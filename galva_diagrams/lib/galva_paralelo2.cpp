///---------------------------------------------------------------------------------------------///
///         	 Galvanostatic simulation code for diagram construction	  	           ///
///---------------------------------------------------------------------------------------------///
///------------------------------------------------------------------------------------------------
/// This simulation code was written to simulate the charging process of a single-particle electrode
/// of a lithium-ion battery. To solve the Fick diffusion equation the Backward-Implicit method was
/// applied. The electrode/electrolyte interface kinetics is simulated by the Butler-Volmer equation
/// using experimental curves for the equilibrium potential.
/// The programm generates a diagram consisting in N (Xi,L) data points. The present codes was
/// parallelized with OpenMP at the level of a point in L for different Xi.
///------------------------------------------------------------------------------------------------

// #include "params_diagram_L.h" //parameters library

#include <limits.h>
#include <math.h>
#include <omp.h> // OpenMP
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

extern "C" void galva(int Npx, int Npt, int NPOINT, int Niso, double Xif, double Xi0, int NXi,
                      double Lf, double L0, int NL, double D, double ks, double T, double Mr,
                      double m, double rho, double Rohm, double Eoff, double *ai, double *bi,
                      double *ci, double *di, double *titaeq, double *res1, double *res2,
                      double *res3) {

  // FILE *archivo, *archivo1;

  // /// In-put
  // #define lee_info_en "isoterma-csaps-galva6.dat"
  // /// Out-put
  // #define out_put "python-OMP-O3-maxthread.dat"

  // /// INITIALIZATION
  // /// STEP--------------------------------------------------------------------------------------
  // // Generation of output file title
  // (archivo = fopen(out_put, "a"));
  // fprintf(archivo, "logL logXi SOC Potential[V]\n");
  // fclose(archivo);

  // Defining simulation parameters
  // int 		omp_get_num_threads(void);
  const int NMOD = Npt / NPOINT; /// Printing Module
  const double F = 96484.5561;   // faraday constant
  const double R = 8.314472;
  const double th = 3600.0;
  const double f = R * T / F;
  const double deltaXi = (Xif - Xi0) / NXi;
  const double deltaL = (Lf - L0) / NL;

  // Diagram parameters
  double logXi[NXi];
  double logL[NL];
  double ii = 0.0;
  for (int i = 0; i < NXi; i++) {
    logXi[i] = Xi0 + deltaXi * ii;
    ii++;
  }
  ii = 0.0;
  for (int i = 0; i < NL; i++) {
    logL[i] = L0 + deltaL * ii;
    ii++;
  }
  int pp = 0;

  int num_threads = omp_get_num_procs();
  omp_set_num_threads(num_threads);
  // printf("nt=%d",num_threads);getchar();
  // int thread_id
/// DIAGRAM LOOP
/// STEP-----------------------------------------------------------------------------------------
#pragma omp parallel
  {
#pragma omp for collapse(2) firstprivate(logXi, logL)
    for (int EL = 0; EL < NL; EL++) { /// L Loop
      for (int XI = 0; XI < NXi; XI++) {
        int thread_id = omp_get_thread_num();
        // printf("id=%d",thread_id);///Xi Loop
        /// Actualization of the parameters
        double L = pow(10, logL[EL]);
        double Xi = pow(10, logXi[XI]);
        double Cr = (ks / Xi) * (ks / Xi) * (th / D);   /// C-rate
        double d = 2.0 * sqrt((L * 2.0 * D * th) / Cr); /// particle diameter, cm
        double S = 4.0 * m / (rho * d);                 /// Surface area, cm2
        // double	Vol=m/rho; 						      	///Volume of
        // active mass, cm3
        double ic =
            -(Cr * 0.5 * d / (2.0 * th)) * (rho / Mr) * F; /// constant current density, A/cm2
        double iR = Rohm * ic * S;                         /// IR drop, A*ohm=V
        double c1 = rho / Mr;
        double iN = 1.0 / (F * D * c1);
        double ttot = 0.5 * 0.5 * d * (rho / Mr) * F / (-ic); /// total time, s CHEQUEAAARRRR
        double Dt = ttot / (Npt - 1.0);                       /// time step, s
        double Dd = 0.5 * d / (Npx - 1.0);                    /// space step, cm

        // Cleaning vectors
        double betaT[Npx], alfaT[Npx], bN[Npx], tita0[Npx], tita1[Npx];

        for (int i = 0; i < Npx; i++) {
          betaT[i] = alfaT[i] = bN[i] = tita0[i] = tita1[i] = 0.0;
        }
        double ii = 0.0;
        double r[Npx];
        for (int i = 0; i < Npx; i++) {
          r[i] = ii * Dd;
          ii++;
        }

        ////Backward implicit parameters and Constant Thomas coefficients
        double Abi = D * Dt / (Dd * Dd);
        double Bbi = D * Dt / (2.0 * Dd);
        double A0bi = 1.0 + (2.0 * Abi);
        alfaT[1] = 2.0 * Abi / A0bi;
        for (int i = 2; i < Npx; i++) {
          alfaT[i] =
              (Abi + (Bbi / (r[i - 1]))) / (A0bi - (Abi - (Bbi / (r[i - 1]))) * alfaT[i - 1]);
        }
        /// Initial Point
        double ti = 0.0;
        for (int i = 0; i < Npx; i++) {
          tita1[i] = titaeq[1];
        }
        double Ei = Eoff + 1.0; // any value just that Ei>Eoff
        double E0 = 0.0;

        int Npot = 0;

        /// TIME LOOP------------------------------------------------------------------------
        while( (Ei >= Eoff) && (Npot < Niso - 1)) {

          double Ai, Bi, Ci, Di, titad, dtitas, E0, i0;

          for (int i = 0; i < Niso; i++) {
            if ((tita1[Npx - 1] >= titaeq[i]) && (tita1[Npx - 1] < titaeq[i + 1])) {
              Ai = ai[i];
              Bi = bi[i];
              Ci = ci[i];
              Di = di[i];
              titad = titaeq[i];
              Npot = i;
              break;
            }
          }
          dtitas = tita1[Npx - 1] - titad;

          // Equilibrium potential calculation 
          E0 = Ai + Bi * dtitas + Ci * dtitas * dtitas + Di * dtitas * dtitas * dtitas;
          i0 = F * c1 * ks * sqrt(tita1[Npx - 1]) * sqrt(1.0 - tita1[Npx - 1]);
          // Potential calculation
          double Ei = E0 + 2.0 * f * asinh(ic / (2.0 * i0));

          // double E0S = E0 + f * log((1 - tita1[Npx - 1]) / tita1[Npx - 1]);
          // double i0 = F * c1 * ks * sqrt(tita1[Npx - 1] * (1.0 - tita1[Npx - 1]));
          // // Potential calculation
          // Ei = E0S + 2.0 * f * asinh(ic / (2.0 * i0));

          /// ACTUALIZATION STEP
          for (int i = 0; i < Npx; i++) {
            tita0[i] = tita1[i];
            betaT[i] = bN[i] = tita1[i] = 0.0;
          }

          // Vector of solutions and Thomas coefficients calculation
          bN[0] = tita0[0];
          bN[Npx - 1] = tita0[Npx - 1] - ((Abi + (Bbi / r[Npx - 1])) * 2 * Dd * (ic * iN));
          for (int i = 1; i < Npx - 1; i++) {
            bN[i] = tita0[i];
          }

          betaT[1] = bN[0] / A0bi;
          for (int i = 2; i < Npx; i++) {
            betaT[i] = (bN[i - 1] + ((Abi - (Bbi / (r[i - 1]))) * betaT[i - 1])) /
                       (A0bi - (Abi - (Bbi / (r[i - 1]))) * alfaT[i - 1]);
          }

          // Concentration calculation
          tita1[Npx - 1] =
              (bN[Npx - 1] + 2.0 * Abi * betaT[Npx - 1]) / (A0bi - 2.0 * Abi * alfaT[Npx - 1]);
          for (int i = 2; i < Npx + 1; i++) {
            tita1[Npx - i] = (alfaT[Npx - (i - 1)] * tita1[Npx - (i - 1)]) + betaT[Npx - (i - 1)];
          }
          ti += Dt; /// time increment
          Npot += 1;
        }           /// End of Time Loop
        /// PRINT POTENTIAL PROFILE POINT AFTER WHILE LOOP ENDS
        double SOC = 0.0;
        for (int i = 0; i < Npx; i++) {
          SOC += tita0[i];
        }
        SOC /= (Npx - 1);

        res1[pp] = logL[EL];
        res2[pp] = logXi[XI];
        res3[pp] = SOC;

        pp++;
      } /// En of Xi loop
    }   /// End of L loop
  }     /// PARALLELIZATION
} ///---------------------------------------------------------------------------------------------------------------------------------------
