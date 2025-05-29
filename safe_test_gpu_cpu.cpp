#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_Array.H>


/*
amrex::Gpu::DeviceVector<T> is a wrapper designed to abstract away the backend:

When CUDA/HIP/SYCL is enabled, it uses GPU device memory.

When not, it falls back to host memory (similar to amrex::Vector<T>).

safely use it with ParallelFor or manually in CPU code.
*/

using namespace amrex;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void loopNaN(int x, int y, int z, int ncomp, const Array4<Real>& mfab_multi_array4D,
    bool* iftrue_device_ptr, bool* ifPrintNANGrid){

    if(amrex::Gpu::isnan(mfab_multi_array4D(x,y,z,ncomp))){
        if(*ifPrintNANGrid){
            printf("mfab(%d,%d,%d,%d) = %f\t", x,y,z,ncomp, mfab_multi_array4D(x,y,z,ncomp));
        }
        *iftrue_device_ptr = true;
    }
}

AMREX_GPU_HOST
bool loopNaN(MultiFab& mfab, bool ifPrintNANGrid){
    auto const & mfab_multi_array4D = mfab.arrays();
    int ncomp = mfab.nComp();
    amrex::Gpu::DeviceScalar<bool> ifPrintNANGrid_device(ifPrintNANGrid); 
    bool iftrue_host = false;   // default return result is "is not NaN"
    amrex::Gpu::DeviceScalar<bool> iftrue_device(false); // Device-side boolean
    bool* iftrue_device_ptr = iftrue_device.dataPtr();
    bool* ifPrintNANGrid_device_ptr = ifPrintNANGrid_device.dataPtr();

    for(int n=0; n<ncomp; n++){
        // reference [&] use instead of [=] value in Lambda expression;
        ParallelFor(mfab, IntVect(0), [=] AMREX_GPU_HOST_DEVICE(int nbx, int x, int y, int z) {
            loopNaN(x,y,z,n,mfab_multi_array4D[nbx],iftrue_device_ptr,ifPrintNANGrid_device_ptr);
        });
    }
    // Copy back the result
    amrex::Gpu::synchronize();
    iftrue_host = iftrue_device.dataValue();

    return iftrue_host;
}
AMREX_GPU_HOST AMREX_FORCE_INLINE
bool MultiFabNANCheck(MultiFab& mfab, bool ifPrintNANGrid, int nstep = -1) {
    if(loopNaN(mfab, ifPrintNANGrid)){
        if(nstep>=0){
            printf("NAN accurs at step %d, stop and check!\n", nstep);
            exit(0);
        }else{
            printf("The MultiFab has NAN elements\n");
            return true;
        }
    }else{
        Print() << "The MultiFab has no NAN elements\n";
        return false;
    }
}


void ComputeCovarianceSimple(const MultiFab& rho, const Geometry& geom, const RealVect& vec_com)
{
    const auto dx = geom.CellSizeArray();
    const auto problo = geom.ProbLoArray();
    const auto probhi = geom.ProbHiArray(); // Real domain bounds, corresponding to the real_box in Geometry;
    IndexType itype = rho.boxArray().ixType();

    // Buffer to hold partial sums for cov matrix (xx, xy, xz, yy, yz, zz)
    amrex::Gpu::DeviceVector<Real> cov_buf_gpu(6, 0.0);
    Real* cov_ptr = cov_buf_gpu.data();

    // Loop over boxes, accumulate partial sums on device buffer
    for (MFIter mfi(rho); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();
        auto const& rho_arr = rho.const_array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            Real m = rho_arr(i,j,k);

            Real x = (i + 0.5 * (1 - itype[0])) * dx[0] + problo[0];
            Real y = (j + 0.5 * (1 - itype[1])) * dx[1] + problo[1];
            Real z = (k + 0.5 * (1 - itype[2])) * dx[2] + problo[2];

            Real dx_ = x - vec_com[0];
            Real dy_ = y - vec_com[1];
            Real dz_ = z - vec_com[2];

            // Atomic add to avoid race conditions:
            amrex::Gpu::Atomic::Add(&cov_ptr[0], m * dx_ * dx_); // xx
            amrex::Gpu::Atomic::Add(&cov_ptr[1], m * dx_ * dy_); // xy
            amrex::Gpu::Atomic::Add(&cov_ptr[2], m * dx_ * dz_); // xz
            amrex::Gpu::Atomic::Add(&cov_ptr[3], m * dy_ * dy_); // yy
            amrex::Gpu::Atomic::Add(&cov_ptr[4], m * dy_ * dz_); // yz
            amrex::Gpu::Atomic::Add(&cov_ptr[5], m * dz_ * dz_); // zz
        });
    }

    // Wait for GPU to finish
    Gpu::synchronize();

    // Copy back to host
    Real h_cov[6];
    Gpu::copy(Gpu::deviceToHost, cov_buf_gpu.data(), cov_buf_gpu.data() + 6, h_cov);

    // Pack into symmetric matrix
    amrex::Array<amrex::Array<Real,3>,3> covar = {{
        {h_cov[0], h_cov[1], h_cov[2]},
        {h_cov[1], h_cov[3], h_cov[4]},
        {h_cov[2], h_cov[4], h_cov[5]}
    }};

    // Print
    amrex::Print() << "Covariance matrix:\n";
    for (int i=0; i<3; ++i) {
        amrex::Print() << covar[i][0] << " " << covar[i][1] << " " << covar[i][2] << "\n";
    }
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    
    #ifdef AMREX_USE_CUDA
    Print() << "Using CUDA Device\n";
    #endif

    int nx = 32;
    IntVect dom_lo(0, 0, 0);
    // ********************  Box Domain Setting, change with the system  ************************
    IntVect dom_hi;
    dom_hi = IntVect(nx-1, nx-1, nx-1);     // for droplet


    const Array<int,3> periodicity({1,1,1});
    Box domain(dom_lo, dom_hi);
    RealBox real_box({0.,0.,0.},{1.,1.,1.});
    Geometry geom(domain, real_box, CoordSys::cartesian, periodicity);
    BoxArray ba(domain);
    // split BoxArray into chunks no larger than "max_grid_size" along a direction
    ba.maxSize(16);
    DistributionMapping dm(ba);
    // need two halo layers for laplacian operator
    int nghost = 0; //2;
    // number of hydrodynamic fields to output
    MultiFab rhof(ba, dm, 1, nghost); rhof.setVal(1.0); // Initialize with some value
    // Center of mass
    RealVect vec_com = {0.5, 0.5, 0.5}; // Assume center of mass is at the center of the domain

    ComputeCovarianceSimple(rhof, geom, vec_com);
    MultiFabNANCheck(rhof, true);

    amrex::Finalize();
}