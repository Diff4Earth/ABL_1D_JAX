# ABL_1D_JAX
WIP materials for the ABL_1D_JAX project as part of the 2025 IGE Jaxathon.

# Contents
- abl_analysis.md: description of abl_analysis fortran code
- ablmod_analysis.md: description of ablmod_analysis fortran code
- par_abl_analysis.md: description of parbal_analysis fortran code
- sbcabl_analysis.md: description of sbcabl_analysis fortran code



# Test Case
In order to validate the first function translation (Hanning 2D filter computation), we extract this function from the ABL code in Fortran (code in `fortran_test_cases`).
We then translate this function in both python and JAX (running either on GPUs and CPUs)

For the test case we compute the Hanning Filter on boundary layer height from ERA5
![](blh_test.png)


# Conclusion from the project

The person doing this translation will need solid knowledge of Fortran and especially NEMO convention and functions. Why people do double loops in (x, y) dimension ordering the indexes like (y_min, y_max, x_min, x_max)...

# ABL in JAX
The set of initial equations in a single-column approach could be write more directly in JAX

There is a set of vertical diffusion equation
<img width="389" alt="VerticalDiffEquation" src="https://github.com/user-attachments/assets/acb9d01f-bb50-4933-8f2e-5a9413a2b0a1" />

TKE closure

