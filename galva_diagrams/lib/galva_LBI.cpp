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

extern "C" void galva(bool frumkin, double g, int Npx, int Npt, int NPOINT, int Niso, double D,
                      double ks, double T, double Mr, double m, double rho, double Rohm,
                      double Eoff, double Qmax, double geo, double logXi, double logL,
                      double SOCperf, double *ai, double *bi, double *ci, double *di,
                      double *titaeq, double *res1, double *res2, double *res3, double *res4) {

  // Defining simulation parameters
  // int 		omp_get_num_threads(void);
  const int NMOD = Npt / NPOINT; /// Printing Module
  const double F = 96484.5561;   // faraday constant
  const double R = 8.314472;
  const double th = 3600.0;
  const double f = R * T / F;

  double L = pow(10, logL);
  double Xi = pow(10, logXi);

  /// Actualization of the parameters

  double Cr = (ks / Xi) * (ks / Xi) * (th / D);   /// C-rate
  double d = 2.0 * sqrt((L * 2.0 * D * th) / Cr); /// particle diameter, cm
  double S = 2.0 * (1 + geo) * m / (rho * d);     /// Surface area, cm2
  // double	Vol=m/rho; 						      	///Volume of
  // active mass, cm3
  double ic = -Cr * Qmax * m / (1000 * S); /// constant current density, A/cm2
  double iR = Rohm * ic * S;               /// IR drop, A*ohm=V
  double c1 = rho / Mr;
  double iN = 1.0 / (F * D * c1);
  //  double ttot = 0.5 * 0.5 * d * (rho / Mr) * F / (-ic); /// total time, s CHEQUEAAARRRR
  double ttot = abs(Qmax * m * 3.6 / (ic * S));
  double NT = Npt;
  double NX = Npx;
  double Dt = ttot / (NT - 1.0);    /// time step, s
  double Dd = 0.5 * d / (NX - 1.0); /// space step, cm

  // Cleaning vectors
  double betaT[Npx], alfaT[Npx], bN[Npx], tita0[Npx], tita1[Npx], r[Npx];

  for (int i = 0; i < Npx; i++) {
    betaT[i] = alfaT[i] = bN[i] = tita0[i] = tita1[i] = 0.0;
    r[i] = i * Dd;
  }

  ////Backward implicit parameters and Constant Thomas coefficients
  double Abi = D * Dt / (Dd * Dd);
  double Bbi = geo * D * Dt / (2.0 * Dd);
  double A0bi = 1.0 + (2.0 * Abi);
  alfaT[1] = 2.0 * Abi / A0bi;
  for (int i = 2; i < Npx; i++) {
    alfaT[i] = (Abi + (Bbi / (r[i - 1]))) / (A0bi - (Abi - (Bbi / (r[i - 1]))) * alfaT[i - 1]);
  }

  /// Initial Point
  switch (frumkin) {
  case true:
    for (int i = 0; i < Npx; i++) {
      tita1[i] = 1e-4;
    }
    break;
  case false:
    for (int i = 0; i < Npx; i++) {
      tita1[i] = titaeq[0];
    }
    break;
  }

  double Ei = Eoff + 1.0; // any value just that Ei>Eoff
  double E0f = 0.0;       // en el caso de frumkin E0 lo debería definir el usuario
  double ti = 0.0;

  int Npot = 0;
  int TP = 0;
  int hh = 0;
  int out = 0;
  double step = 1e-4;

  /// TIME LOOP------------------------------------------------------------------------
  switch (frumkin) {
  case true:
    while (Ei > Eoff) {
      /// POTENTIAL CALCULATION STEP
      // Search range of experimental points where superficial concentration (tita1) belongs
      double i0 = F * c1 * ks * sqrt(tita1[Npx - 1] * (1.0 - tita1[Npx - 1]));
      double Eg = E0f + f * (g * (0.5 - tita1[Npx - 1]) +
                             log((1.0 - tita1[Npx - 1]) / tita1[Npx - 1])); /// Frumkin

      // Potential calculation
      Ei = Eg + 2.0 * f * asinh(ic / (2.0 * i0));

      /// PRINT POTENTIAL PROFILE POINT
      double SOC = 0.0;
      for (int i = 0; i < Npx; i++) {
        SOC += tita1[i];
      }
      SOC /= (NX - 1);

      if (TP % NMOD == 0) {
        res1[hh] = SOC;
        res2[hh] = Ei;
        hh++;
      }

      if ((SOC > SOCperf - step) && (SOC < SOCperf + step)) {
        if (out == 0) {
          for (int i = 0; i < Npx; i++) {
            res3[i] = r[i] / (d * 0.5);
            res4[i] = tita1[i];
          }
          out++;
        }
      }

      /// ACTUALIZATION STEP
      for (int i = 0; i < Npx; i++) {
        tita0[i] = tita1[i];
        betaT[i] = bN[i] = tita1[i] = 0.0;
      }

      // Vector of solutions and Thomas coefficients calculation
      bN[0] = tita0[0];
      bN[Npx - 1] = tita0[Npx - 1] - ((Abi + (Bbi / r[Npx - 1])) * 2.0 * Dd * (ic * iN));
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
      ti += Dt;
      TP++; /// time increment
    }
    break;

  case false:
    while (Ei > Eoff) {
      /// POTENTIAL CALCULATION STEP
      // Search range of experimental points where superficial concentration (tita1) belongs
      double Ai, Bi, Ci, Di, titad;
      for (int i = 0; i < Niso; i++) {
        if ((tita1[Npx - 1] >= titaeq[i]) && (tita1[Npx - 1] < titaeq[i + 1])) {
          Ai = ai[i];
          Bi = bi[i];
          Ci = ci[i];
          Di = di[i];
          titad = titaeq[i];
          break;
        }
      }
      double dtitas = tita1[Npx - 1] - titad;

      // Equilibrium potential calculation
      double E0 = Ai + Bi * dtitas + Ci * dtitas * dtitas + Di * dtitas * dtitas * dtitas;
      double i0 = F * c1 * ks * sqrt(tita1[Npx - 1] * (1.0 - tita1[Npx - 1]));
      // Potential calculation
      Ei = E0 + 2.0 * f * asinh(ic / (2.0 * i0));

      /// PRINT POTENTIAL PROFILE POINT
      double SOC = 0.0;
      for (int i = 0; i < Npx; i++) {
        SOC += tita1[i];
      }
      SOC /= (NX - 1);

      if (TP % NMOD == 0) {
        res1[hh] = SOC;
        res2[hh] = Ei;
        hh++;
      }

      if ((SOC > SOCperf - step) && (SOC < SOCperf + step)) {
        if (out == 0) {
          for (int i = 0; i < Npx; i++) {
            res3[i] = r[i] / (d * 0.5);
            res4[i] = tita1[i];
          }
          out++;
        }
      }

      /// ACTUALIZATION STEP
      for (int i = 0; i < Npx; i++) {
        tita0[i] = tita1[i];
        betaT[i] = bN[i] = tita1[i] = 0.0;
      }

      // Vector of solutions and Thomas coefficients calculation
      bN[0] = tita0[0];
      bN[Npx - 1] = tita0[Npx - 1] - ((Abi + (Bbi / r[Npx - 1])) * 2.0 * Dd * (ic * iN));
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
      ti += Dt;
      TP++; /// time increment
    }
    break;
  }
} //---------------------------------------------------------------------------------------------------------------------------------------
