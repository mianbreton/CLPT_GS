/* ============================================================ *
 * AUTHOR:      Antoine Rocher       2019		        *
 * CONTRIBUTOR: Michel-Andrès Breton 2020		        *
 * 							        *
 * Code for computing Gaussian Streaming		        *
 * ============================================================ */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_sf_legendre.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include <time.h>

#define r_max 250
/************************************************************************************************************************************************************************/
/************************************************************************************************************************************************************************/

                        //VARIABLE GLOBALE
const double y_spanning = 50;						        // Spanning for streaming model
gsl_interp_accel *acc[4];
gsl_spline *spline[4];
double				fg,						// growth factor
			 	smin, 						// Minimum of s range
			 	smax, 						// Maximum of s range
			 	f1,						// First Bias factor CLPT (related to the linear Eulerian bias as b = 1 + f1)
			 	f2,						// Second Bias factor CLPT
				sig_shift,					// Sigma^2 for FoG effect
				alpha_perp_AP,					// alpha_perp for AP test
				alpha_para_AP;					// alpha_para for AP test
int nbins;								        // Number of bin in s
double Xi_x[r_max], Xi_f0[r_max], Xi_f1[r_max], Xi_f2[r_max], Xi_f1_2[r_max], Xi_f1_f2[r_max], Xi_f2_2[r_max],
       V12_x[r_max], V12_f0[r_max], V12_f1[r_max], V12_f2[r_max], V12_f1_2[r_max], V12_f1_f2[r_max],
       S_x[r_max], S_par_f0[r_max], S_par_f1[r_max], S_par_f2[r_max], S_par_f1_2[r_max], S_per_f0[r_max], S_per_f1[r_max], S_per_f2[r_max], S_per_f1_2[r_max];
struct my_f_params { double a; double b; };
struct my_f_params3 { double a; double b; double c;};

/*********************************************************************************************************************************************************************/
/*******************\\ Gauss-Legendre integral quadrature\\***********************************************************************************************************/
																															
static const double x[] = {
    1.56289844215430828714e-02,    4.68716824215916316162e-02,
    7.80685828134366366918e-02,    1.09189203580061115002e-01,
    1.40203137236113973212e-01,    1.71080080538603274883e-01,
    2.01789864095735997236e-01,    2.32302481844973969643e-01,
    2.62588120371503479163e-01,    2.92617188038471964730e-01,
    3.22360343900529151720e-01,    3.51788526372421720979e-01,
    3.80872981624629956772e-01,    4.09585291678301542532e-01,
    4.37897402172031513100e-01,    4.65781649773358042251e-01,
    4.93210789208190933576e-01,    5.20158019881763056670e-01,
    5.46597012065094167460e-01,    5.72501932621381191292e-01,
    5.97847470247178721259e-01,    6.22608860203707771585e-01,
    6.46761908514129279840e-01,    6.70283015603141015784e-01,
    6.93149199355801965946e-01,    7.15338117573056446485e-01,
    7.36828089802020705530e-01,    7.57598118519707176062e-01,
    7.77627909649495475605e-01,    7.96897892390314476375e-01,
    8.15389238339176254384e-01,    8.33083879888400823522e-01,
    8.49964527879591284320e-01,    8.66014688497164623416e-01,
    8.81218679385018415547e-01,    8.95561644970726986709e-01,
    9.09029570982529690453e-01,    9.21609298145333952679e-01,
    9.33288535043079545942e-01,    9.44055870136255977955e-01,
    9.53900782925491742847e-01,    9.62813654255815527284e-01,
    9.70785775763706331929e-01,    9.77809358486918288561e-01,
    9.83877540706057015509e-01,    9.88984395242991747997e-01,
    9.93124937037443459632e-01,    9.96295134733125149166e-01,
    9.98491950639595818382e-01,    9.99713726773441233703e-01
};

static const double A[] = {
    3.12554234538633569472e-02,    3.12248842548493577326e-02,
    3.11638356962099067834e-02,    3.10723374275665165874e-02,
    3.09504788504909882337e-02,    3.07983790311525904274e-02,
    3.06161865839804484966e-02,    3.04040795264548200160e-02,
    3.01622651051691449196e-02,    2.98909795933328309169e-02,
    2.95904880599126425122e-02,    2.92610841106382766198e-02,
    2.89030896011252031353e-02,    2.85168543223950979908e-02,
    2.81027556591011733175e-02,    2.76611982207923882944e-02,
    2.71926134465768801373e-02,    2.66974591835709626611e-02,
    2.61762192395456763420e-02,    2.56294029102081160751e-02,
    2.50575444815795897034e-02,    2.44612027079570527207e-02,
    2.38409602659682059633e-02,    2.31974231852541216230e-02,
    2.25312202563362727021e-02,    2.18430024162473863146e-02,
    2.11334421125276415432e-02,    2.04032326462094327666e-02,
    1.96530874944353058650e-02,    1.88837396133749045537e-02,
    1.80959407221281166640e-02,    1.72904605683235824399e-02,
    1.64680861761452126430e-02,    1.56296210775460027242e-02,
    1.47758845274413017686e-02,    1.39077107037187726882e-02,
    1.30259478929715422855e-02,    1.21314576629794974079e-02,
    1.12251140231859771176e-02,    1.03078025748689695861e-02,
    9.38041965369445795116e-03,    8.44387146966897140266e-03,
    7.49907325546471157895e-03,    6.54694845084532276405e-03,
    5.58842800386551515727e-03,    4.62445006342211935096e-03,
    3.65596120132637518238e-03,    2.68392537155348241939e-03,
    1.70939265351810523958e-03,    7.34634490505671730396e-04
};

#define NUM_OF_POSITIVE_ZEROS  sizeof(x) / sizeof(double)
#define NUM_OF_ZEROS           NUM_OF_POSITIVE_ZEROS+NUM_OF_POSITIVE_ZEROS

double Gauss_Legendre_Integration2_100pts(double a, double b, double (*f)(double, void *), void *prms)
{
   double integral = 0.0; 
   double c = 0.5 * (b - a);
   double d = 0.5 * (b + a);
   double dum;
   const double *px = &x[NUM_OF_POSITIVE_ZEROS - 1];
   const double *pA = &A[NUM_OF_POSITIVE_ZEROS - 1];
   for (; px >= x; pA--, px--) {
      dum = c * *px;
      integral += *pA * ( (*f)(d - dum,prms) + (*f)(d + dum,prms) );
   }

   return c * integral;
}

void Gauss_Legendre_Integration2_100pts_array(double a, double b, void (*f)(double, void *, double []), void *prms, double result[], int n)
{
   int i;
   double integral[n];
   for (i=0; i<n; i++)
	integral[i] = 0;	   		 
    double c = 0.5 * (b - a);
    double d = 0.5 * (b + a);
    double dum;
    const double *px = &x[NUM_OF_POSITIVE_ZEROS - 1];
    const double *pA = &A[NUM_OF_POSITIVE_ZEROS - 1];
    double outi[n], outs[n];


    for (; px >= x; pA--, px--) {
	dum = c * *px;
	(*f)(d - dum, prms, outi);
	(*f)(d + dum, prms, outs);
	for (i=0; i<n; i++)
	    integral[i] += *pA * (outi[i] + outs[i]);
    }
    for (i=0; i<n; i++)
	result[i] = c*integral[i];
}

/***********************************************************************************************************************************************************************/
/********************************\\ Interpolation Function \\***********************************************************************************************************/

double Xi_R_CLPT(double s)
{
	if (s>= 1. && s<=r_max)	return	 gsl_spline_eval (spline[0], s, acc[0]);
	else if (s < 1.)	return	 gsl_spline_eval (spline[0], 1., acc[0]);
	else if (s > r_max)	return	 gsl_spline_eval (spline[0], r_max, acc[0]);
        else	return 0;
}

double V12_CLPT(double s)
{	if (s>= 1. && s<=r_max)	return 	gsl_spline_eval (spline[1], s, acc[1]);
	else if (s < 1.)	return 	gsl_spline_eval (spline[1], 1., acc[1]);
	else if (s > r_max)	return 	gsl_spline_eval (spline[1], r_max, acc[1]);
        else return 0;
}

double Sig_par_CLPT(double s)
{
	if (s>= 1. && s<= r_max)return 	gsl_spline_eval (spline[2], s, acc[2]);
	else if (s < 1.) 	return 	gsl_spline_eval (spline[2], 1., acc[2]);
	else if (s > r_max) 	return 	gsl_spline_eval (spline[2], r_max, acc[2]);
        else return 0;
}

double Sig_per_CLPT(double s)
{
	if (s>= 1. && s<=r_max)	return 	gsl_spline_eval (spline[3], s, acc[3]);
	else if (s < 1.)	return 	gsl_spline_eval (spline[3], 1., acc[3]);
	else if (s > r_max)	return 	gsl_spline_eval (spline[3], r_max, acc[3]);
        else    return 0;
}

/**********************************************************************************************************************************************************************/
/*************\\ Gaussian Function\\***********************************************************************************************************************************/
     
double gauss (double x, double moy, double var)
{
    return 1./(sqrt(2.*M_PI*var))*exp(-0.5*(x-moy)*(x-moy)/var);
}

/**********************************************************************************************************************************************************************/
/*************\\Decomposition of s and r\\*****************************************************************************************************************************/
                                                                                                                              
double spar (double s, double mu_s)
{
    return s*mu_s;
}

double rperp (double s, double spar) // rp = sp (perp)
{
    return sqrt(s*s-spar*spar);
}

double r_real(double rp, double rpar)
{
    return sqrt(rp*rp+rpar*rpar);   
}

double mu_r(double rpar, double r)
{
    return rpar/r;
}

/********Sigma_12(mu,r)***************************************************************************************************************************************************/

double Sigma12_CLPT (double mu, double r)							// from CLPT prediction
{
    return (mu*mu*Sig_par_CLPT(r)+(1.-mu*mu)*Sig_per_CLPT(r))/(1.+Xi_R_CLPT(r));
}

double Sigma12_CLPT_with_Xir (double Xi_r, double mu, double r)							// from CLPT prediction
{
    return (mu*mu*Sig_par_CLPT(r)+(1.-mu*mu)*Sig_per_CLPT(r))/(1.+Xi_r);
}


/*********************************************************************************************************************************************************************/
/****************\\Correlation function in z-space for CLPT prediction\\**********************************************************************************************/

double fXis_CLPT (double y, void *p)
{
    struct my_f_params * params = (struct my_f_params *)p;
    double spara = (params->a);
    double rperp = (params->b);
    double r = r_real(rperp, y);  
    double Xi_r = Xi_R_CLPT(r);                      					
    double v = fg*V12_CLPT(r)/(1.+Xi_r);
    double mu_r = y/r;
    double x = spara-y;
    double moy = mu_r*v;
    double var = fg*fg*Sigma12_CLPT_with_Xir(Xi_r, mu_r,r) + sig_shift;
	
    if (var>0){
	return (1.+Xi_r)*gauss(x, moy, var); 
    }
    else
	return 0;
}

double Xis_CLPT (double sperp, double spara)   
{
    struct my_f_params params = {spara, sperp};  
    double result = Gauss_Legendre_Integration2_100pts(spara - y_spanning, spara + y_spanning, &fXis_CLPT, &params);
    return result - 1.;	
}

/********************************************************************************************************************************************************************/
/****************\\Legendre Multipole with CLPT prediction\\*********************************************************************************************************/
														
void fmultipole_CLPT(double mu, void *prms, double result[]) 
{
    struct my_f_params3 * params = (struct my_f_params3 *)prms;
    double s = (params->a);
    double alpha_perp = (params->b);
    double alpha_para = (params->c);

    double rperp = alpha_perp*s*sqrt(1.-mu*mu);
    double spara = alpha_para*s*mu;												// Decomposition of S in spar (=spi) and sp=rp
    double Xi_s= Xis_CLPT(rperp,spara);
    result[0] = Xi_s*gsl_sf_legendre_Pl(0,mu);
    result[1] = Xi_s*gsl_sf_legendre_Pl(2,mu);
    result[2] = Xi_s*gsl_sf_legendre_Pl(4,mu);
}

void multipole_CLPT(double s, double alpha_perp_AP, double alpha_para_AP, double result[])   
{
    struct my_f_params3 p = {s, alpha_perp_AP, alpha_para_AP}; 
    Gauss_Legendre_Integration2_100pts_array(0, 1, &fmultipole_CLPT, &p, result, 3);
}

/************************************************************************************************************************************************************************/
/***********\\ Interpolation \\*****************************************************************************************************************************/

void interpole(int n, char ficher[100],int vmax)
{
	double T_x[vmax];
	double T_y[vmax];
	FILE* f;
	f =fopen(ficher, "r");
	for(int i=0; i < vmax; i++) 
	    fscanf(f, "%lf %lf\n", &T_x[i], &T_y[i]);
        fclose(f);

	acc[n] = gsl_interp_accel_alloc ();
        spline[n] = gsl_spline_alloc(gsl_interp_cspline, vmax);
        gsl_spline_init (spline[n], T_x, T_y, vmax);
}

void read_Xi(char* input_dir)
{
	FILE *fxi;
    	char dir_tmp[BUFSIZ];
    	strcpy(dir_tmp, input_dir);
	char *name = "/Xi_r_CLPT.dat";
	strcat(dir_tmp, name);
	fxi = fopen(dir_tmp, "r");
	for(int i=0; i < r_max; i++){
	    fscanf(fxi, "%lf %lf %lf %lf %lf %lf %lf\n",&Xi_x[i], &Xi_f0[i], &Xi_f1[i], &Xi_f2[i], &Xi_f1_2[i], &Xi_f1_f2[i], &Xi_f2_2[i]); 
	}
        fclose(fxi);
}

void read_V12(char* input_dir)
{
	FILE *fv12;
    	char dir_tmp[BUFSIZ];
    	strcpy(dir_tmp, input_dir);
	char *name = "/V_12_CLPT.dat";
	strcat(dir_tmp, name);
	
	fv12 = fopen(dir_tmp, "r");
	for(int i=0; i < r_max; i++){
	    fscanf(fv12, "%lf %lf %lf %lf %lf %lf\n",&V12_x[i], &V12_f0[i], &V12_f1[i], &V12_f2[i], &V12_f1_2[i], &V12_f1_f2[i]);
	}
        fclose(fv12);
}

void read_sigma(char* input_dir)
{
	FILE* fsig;
    	char dir_tmp[BUFSIZ];
    	strcpy(dir_tmp, input_dir);
	char *name = "/Sigma_12_CLPT.dat";
	strcat(dir_tmp, name);
	
   	fsig = fopen(dir_tmp, "r");
	for(int i=0; i<r_max; i++){
	    fscanf(fsig, "%lf %lf %lf %lf %lf %lf %lf %lf %lf\n", &S_x[i], &S_par_f0[i], &S_par_f1[i], &S_par_f2[i], &S_par_f1_2[i], &S_per_f0[i], &S_per_f1[i], &S_per_f2[i], &S_per_f1_2[i]);
	}
        fclose(fsig);
}


void interpole_Xi()
{
	double Xi_all[r_max];
	for(int i=0; i < r_max; i++){
	    Xi_all[i] = Xi_f0[i] + f1*Xi_f1[i] + f2*Xi_f2[i] + f1*f1*Xi_f1_2[i] + f1*f2*Xi_f1_f2[i] + f2*f2*Xi_f2_2[i];
	}
	acc[0] = gsl_interp_accel_alloc ();
	spline[0] = gsl_spline_alloc(gsl_interp_cspline, r_max);
	gsl_spline_init (spline[0], Xi_x, Xi_all, r_max);
}

void interpole_V12()
{

	double V12_all[r_max];
	for(int i=0; i < r_max; i++){
	    V12_all[i] = V12_f0[i] + f1*V12_f1[i] + f2*V12_f2[i] + f1*f1*V12_f1_2[i] + f1*f2*V12_f1_f2[i];
	}
	acc[1] = gsl_interp_accel_alloc ();
	spline[1] = gsl_spline_alloc(gsl_interp_cspline, r_max);
	gsl_spline_init (spline[1], V12_x, V12_all, r_max);
}


void interpole_sigma()
{	
	double S_par_all[r_max], S_per_all[r_max];

	for(int i=0; i<r_max; i++){
	    S_par_all[i] = S_par_f0[i] + f1*S_par_f1[i] + f2*S_par_f2[i] + f1*f1*S_par_f1_2[i];
	    S_per_all[i] = S_per_f0[i] + f1*S_per_f1[i] + f2*S_per_f2[i] + f1*f1*S_per_f1_2[i];
	}

	acc[2] = gsl_interp_accel_alloc ();
	spline[2] = gsl_spline_alloc(gsl_interp_cspline, r_max);
	gsl_spline_init (spline[2], S_x, S_par_all, r_max);

	acc[3] = gsl_interp_accel_alloc ();
	spline[3] = gsl_spline_alloc(gsl_interp_cspline, r_max);
	gsl_spline_init (spline[3], S_x, S_per_all, r_max);
}

void read_moments(char* input_dir)
{
    read_Xi(input_dir);
    read_V12(input_dir);
    read_sigma(input_dir);	
}

void interpole_moments()
{
    interpole_Xi();
    interpole_V12();
    interpole_sigma();	
}

/************************************************************************************************************************************************************************/
/***********\\ Compute multipoles CLPT \\*****************************************************************************************************************************/

void compute_multipoles_CLPT(double in_smin, double in_smax, double in_nbins, double out[], double in_fg, double in_f1, double in_f2, double in_sig_shift, double in_alpha_perp_AP, double in_alpha_para_AP)
{
    smin = in_smin;
    smax = in_smax;
    fg = in_fg;
    f1 = in_f1;
    f2 = in_f2;
    sig_shift = in_sig_shift;
    alpha_perp_AP = in_alpha_perp_AP;
    alpha_para_AP = in_alpha_para_AP;
    nbins = in_nbins;

    double ds = (smax - smin)/(nbins-1);

    //clock_t begin = clock();
    double out_tmp[3];
    // Interpole
    interpole_moments();
    // Compute
    for(int i=0; i < nbins; i++){ 
	double spi = smin + i*ds;  
	multipole_CLPT(spi, alpha_perp_AP, alpha_para_AP, out_tmp);
	out[i] = out_tmp[0];
	out[nbins + i] = 5*out_tmp[1];   
	out[2*nbins + i] = 9*out_tmp[2]; 	                 
    }
    //clock_t end = clock();
    //double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    //printf("Time spent : %fs\n", time_spent);

}

/************************************************************************************************************************************************************************/
/***********\\ Write multipoles CLPT \\*****************************************************************************************************************************/

void write_multipoles_CLPT(FILE* par)
{

    FILE *fi;
    char input_dir[BUFSIZ];
	
// Check parameter file
    if (par!=0) EXIT_SUCCESS;											
    else {
	printf("Wrong parameter file\n"); 
    }

// Read parameter file

    fscanf(par,"%*s %lf\n", &smin);
    fscanf(par,"%*s %lf\n", &smax);
    fscanf(par,"%*s %d\n", &nbins);
    fscanf(par,"%*s %lf\n", &fg);
    fscanf(par,"%*s %lf\n", &f1);
    fscanf(par,"%*s %lf\n", &f2);
    fscanf(par,"%*s %lf\n", &sig_shift);
    fscanf(par,"%*s %lf\n", &alpha_perp_AP);
    fscanf(par,"%*s %lf\n", &alpha_para_AP);
    fscanf(par,"%*s %s\n", input_dir);

// Get distributions
    read_moments(input_dir);
	
// Compute multipoles
    double out[3*nbins];
    compute_multipoles_CLPT(smin, smax, nbins, out, fg, f1, f2, sig_shift, alpha_perp_AP, alpha_para_AP);

// Write multipoles
    char dir_tmp[BUFSIZ];
    strcpy(dir_tmp, input_dir);
    char* name = "/multipoles_CLPT.dat";
    strcat(dir_tmp, name);
    fi = fopen(dir_tmp,"w+");

    double ds = (smax - smin)/(nbins-1);
    for(int i=0; i < nbins; i++){ 
	double spi = smin + i*ds;  
	fprintf(fi,"%lf %.13le %.13le %.13le\n", spi, out[i], out[nbins+i], out[2*nbins+i]);
    }

    fclose(fi);
}


/******************************************************************************************************************************************************************/
/***************\\MAIN FUNCTION\\**********************************************************************************************************************************/
int main (int argc, char *argv[])
{	  

    FILE *par;
    par = fopen(argv[1],"r");  

    printf("Compute RSD multipoles with CLPT-GS\n");
    write_multipoles_CLPT(par);

    for (int i=0; i<4; i++){
    	gsl_spline_free (spline[i]);
   	gsl_interp_accel_free (acc[i]);
    }
   		
    return 0;
}



