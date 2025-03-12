# ABL_1D_JAX
WIP materials for the ABL_1D_JAX project as part of the 2025 IGE Jaxathon.

# Contents
- abl_analysis.md: description of abl_analysis fortran code
- ablmod_analysis.md: description of ablmod_analysis fortran code
- par_abl_analysis.md: description of parbal_analysis fortran code
- sbcabl_analysis.md: description of sbcabl_analysis fortran code

# Module description


# Test Case
In order to validate the first function translation (Hanning 2D filter computation), we extract this function from the ABL code in Fortran (code in `fortran_test_cases`).
We then translate this function in both python and JAX (running either on GPUs and CPUs)

For the test case we compute the Hanning Filter on boundary layer height from ERA5
![](blh_test.png)
