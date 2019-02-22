#include <iostream>
#include <fstream>
#include <math.h>
#include <cstdlib>
#include <string>
#include "eigen3/Eigen/Dense"
#include <math.h>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using namespace std;

const unsigned int SEED = 42;
const unsigned int NUM_SAMPLES = 100000;

//Created by Ben Penwell and Adam Landis
//Pattern Recognition, Project 1
//Feb. 12, 2019
float ranf();
float box_muller(float m, float s);
void generatePairs(float mean, float variance, double array[][2]);
void genSamples(VectorXd mu_i, 
				MatrixXd sigma_i, 
				unsigned numDimensions, 
				string   filename,
				unsigned numSamples = NUM_SAMPLES);
void useBayesianClassifier(string dataFile);
MatrixXd disriminantfunction_Case1_G1(MatrixXd x, MatrixXd mu, float sd, float prior);
MatrixXd linearDiscFunc_case1(MatrixXd x, MatrixXd mu, float sd, float prior);
MatrixXd quadraticDiscFunc_case3(MatrixXd x, MatrixXd mu, MatrixXd sigma, float prior);
void runData(int passInput, string file_G1, string file_G2, VectorXd xVector, MatrixXd mu_G1, MatrixXd mu_G2, MatrixXd sigma_G1, MatrixXd sigma_G2, float prior_G1, float prior_G2);
MatrixXd kBound(float beta, MatrixXd mu_1, MatrixXd mu_2, MatrixXd sigma_1, MatrixXd sigma_2);

int main()
{
	string outputFile;
	float mean, var;
	double array[NUM_SAMPLES][2];
	string input;
	srand(SEED);

	// number of dimensions for feature vector for each class
	unsigned dim = 2;

	// the filenames for class 1 and class 2
	string filename_1 = "mean1_var1";
	string filename_2 = "mean4_var1";
	string filename_3 = "mean4_var4,8";


	// the prior probabilities for class 1 (P(w_1)) and class 2 (P(w_2))
	float pw_1 = 0.2;
	float pw_2 = 0.8;

	// mean matrix for class 1
	VectorXd mu_1(dim);
	mu_1(0) = 1.0;
	mu_1(1) = 1.0;

	// covariance matrix for class 1
	MatrixXd sigma_1(dim, dim);
	sigma_1(0, 0) = 1.0;
	sigma_1(1, 0) = 0.0;
	sigma_1(0, 1) = 0.0;
	sigma_1(1, 1) = 1.0;

	// mean matrix for class 2
	VectorXd mu_2(dim);
	mu_2(0) = 4.0;
	mu_2(1) = 4.0;

	// covariance matrix for class 2
	MatrixXd sigma_2(dim, dim);
	sigma_2(0, 0) = 1.0;
	sigma_2(1, 0) = 0.0;
	sigma_2(0, 1) = 0.0;
	sigma_2(1, 1) = 1.0;
    
    VectorXd mu_3(dim);
    mu_3(0) = 4.0;
    mu_3(1) = 4.0;
    
    MatrixXd sigma_3(dim,dim);
    sigma_3(0, 0) = 4.0;
    sigma_3(0, 1) = 0.0;
    sigma_3(1, 0) = 0.0;
    sigma_3(1, 1) = 8.0;

	while (input != "-1")
	{
		cout << endl
		     << "+==============================================================+\n"
			 << "|Select  1 to generate new datapoints for part 1               |\n"
		     << "|Select  2 to run data on part 1 data                          |\n"
             << "|Select  3 to generate new datapoints for part 2               |\n"
		     << "|Select  4 to run data on part 2 data                          |\n"
		     << "|Select  5 to calculate Bhattacharyya bound for part 1         |\n"
		     << "|Select  6 to calculate Bhattacharyya bound for part 2 & 3     |\n"
		     << "|Select  7 to run part 2 data with minimum-distance classifier |\n"
		     << "|Select -1 to exit                                             |\n"
		     << "+==============================================================+\n"
		     << endl
		     << "Choice: ";

		cin >> input;

		cout << endl;

		if (input == "1")
		{
			genSamples(mu_1, sigma_1, dim, filename_1);
			genSamples(mu_2, sigma_2, dim, filename_2);
		}
		else if (input == "2")
		{
			VectorXd xVector(dim, 1);

			runData(2, filename_1, filename_2, xVector, mu_1, mu_2, sigma_1, sigma_2, pw_1, pw_2);
		}
		else if (input == "3")
		{	
			genSamples(mu_1, sigma_1, dim, filename_1);
			genSamples(mu_3, sigma_3, dim, filename_3);

		}
		else if (input == "4")
		{
			VectorXd xVector(dim, 1);

			runData(4, filename_1, filename_3, xVector, mu_1, mu_3, sigma_1, sigma_3, pw_1, pw_2);

		}
		else if (input == "5")
		{
			float beta = 0.5;
			MatrixXd returnValue = kBound(beta, mu_1, mu_2, sigma_1, sigma_2);
			cout << "Function K(beta = " << beta << ") returns: " << returnValue(0,0) << endl;
		}
		else if (input == "6")
		{
			float beta = 0.5;
			MatrixXd returnValue = kBound(beta, mu_1, mu_3, sigma_1, sigma_3);
			cout << "Function K(beta = " << beta << ") returns: " << returnValue(0,0) << endl;
		}
		else if (input != "-1")
		{
			cout << "\"" << input << "\" is not a valid command" << endl;
		}

	}	
}

/**
 * @brief      Passes the input to gain context to which discriminant function to run
 *
 * @param[in]  passInput  The passed input
 * @param[in]  file_G1    The file for discriminant g1
 * @param[in]  file_G2    The file for discriminant g2
 * @param[in]  xVector    The x vector
 * @param[in]  mu_G1      The mu for discriminant g1
 * @param[in]  mu_G2      The mu for discriminant g2
 * @param[in]  sigma_G1   The sigma for discriminant g1
 * @param[in]  sigma_G2   The sigma for discriminant g2
 * @param[in]  prior_G1   The prior for discriminant g1
 * @param[in]  prior_G2   The prior for discriminant g2
 */
void runData(int passInput, string file_G1, string file_G2, VectorXd xVector, MatrixXd mu_G1, MatrixXd mu_G2, MatrixXd sigma_G1, MatrixXd sigma_G2, float prior_G1, float prior_G2)
{
	// keep track of how many are classified to 
	// dataset G1 (mean=1,var=1) vs dataset G2 (mean=4,var=1)
	
	ifstream fin_G1;
	fin_G1.open(file_G1.c_str());
	ifstream fin_G2;
	fin_G2.open(file_G2.c_str());

	int missClassified = 0;

	float x, y;

	int classifiedAs_i = 0;
	int classifiedAs_j = 0;

	MatrixXd g1Value;
	MatrixXd g2Value;

	cout << "Running first dataset (" << file_G1 << "):\n";
	cout << "(prior_1 = " << prior_G1 << " | prior_2 = " << prior_G2 << ")" << endl;

	while (!fin_G1.eof())
	{
		fin_G1 >> x >> y;
		xVector(0,0) = x;
		xVector(1,0) = y;

		//g1Value & g2Value returns a 1-D array
		if(passInput == 4)
		{
			g1Value = quadraticDiscFunc_case3(xVector, mu_G1, sigma_G1, prior_G1);
			g2Value = quadraticDiscFunc_case3(xVector, mu_G2, sigma_G2, prior_G2);
		}
		else if( passInput == 2)
		{
			g1Value = linearDiscFunc_case1(xVector, mu_G1, 1.0, prior_G1);
			g2Value = linearDiscFunc_case1(xVector, mu_G2, 1.0, prior_G2);
		}

		float temp = g1Value(0, 0) - g2Value(0, 0);

		if (temp >= 0)
		{
			classifiedAs_i++; 
		}
		else
		{
			classifiedAs_j++;
		}

	}
	fin_G1.close();

	cout << "Results: G(x) >= 0 (Decide x [Correctly identified]): " 
		 << classifiedAs_i 
		 << ". G(x) < 0 (Decide y [Incorrectly identified]): " 
		 << classifiedAs_j 
		 << ".\n\n";

	// keep track of how many are classified to 
	// dataset G1 (mean=1,var=1) vs dataset G2 (mean=4,var=1)
 	missClassified += classifiedAs_j;
	classifiedAs_i = 0;
	classifiedAs_j = 0;

	cout << "Running second dataset (" << file_G2 << "):\n\n";
	
	while (!fin_G2.eof())
	{
		fin_G2 >> x >> y;
		xVector(0, 0) = x;
		xVector(1, 0) = y;

		//g1Value & g2Value returns a 1-D array
		if(passInput == 4)
		{
			g1Value = quadraticDiscFunc_case3(xVector, mu_G1, sigma_G1, prior_G1);
			g2Value = quadraticDiscFunc_case3(xVector, mu_G2, sigma_G2, prior_G2);
		}
		else if( passInput == 2)
		{
			g1Value = linearDiscFunc_case1(xVector, mu_G1, 1.0, prior_G1);
			g2Value = linearDiscFunc_case1(xVector, mu_G2, 1.0, prior_G2);
		}
		float temp = g1Value(0, 0) - g2Value(0, 0);

		if (temp >= 0)
		{
			classifiedAs_i++; 
		}
		else
		{
			classifiedAs_j++;
		}

	}
	fin_G2.close();

	cout << "Results: G(x) >= 0 (Decide x [Incorrectly identified]): " 
		 << classifiedAs_i 
		 << ". G(x) < 0 (Decide y [Correctly identified]): " 
		 << classifiedAs_j 
		 << ".\n";
	
	missClassified += classifiedAs_i;

	cout << "Total number of misclassified datapoints: " << missClassified << endl;

}

double ranf(double m)
{
	return (m * rand() / (double)RAND_MAX);
}

//This function was developed by Dr. Everett (Skip) F. Carter J., 
//and all credit for this functionality goes to him.
float box_muller(float m, float s)	/* normal random variate generator */
{				        /* mean m, standard deviation s */
	float x1, x2, w, y1;
	static float y2;
	static int use_last = 0;

	if (use_last)		        /* use value from previous call */
	{
		y1 = y2;
		use_last = 0;
	}
	else
	{
		do {
			x1 = 2.0 * ranf(1) - 1.0;
			x2 = 2.0 * ranf(1) - 1.0;
			w = x1 * x1 + x2 * x2;
		} while ( w >= 1.0 );

		w = sqrt( (-2.0 * log( w ) ) / w );
		y1 = x1 * w;
		y2 = x2 * w;
		use_last = 1;
	}

	return( m + y1 * s );
}

/**
 * @brief      Generates random gaussian samples from a given mean vector, 
 * 			   covariance matrix, and number of dimensions of the feature vector
 *
 * @param[in]  mu_i           The mean vector
 * @param[in]  sigma_i        The covariance matrix
 * @param[in]  numDimensions  The number of dimensions
 * @param[in]  filename       The filename of the file to save the samples to
 * @param[in]  numSamples     The number samples to generate
 * 
 * @return     None
 */
void genSamples(VectorXd mu_i, 
				MatrixXd sigma_i, 
				unsigned numDimensions, 
				string   filename,
				unsigned numSamples)
{
    ofstream fout(filename.c_str());

    for (int n = 0; n < numSamples; n++)
    {
        for (int d = 0; d < numDimensions; d++)
        {
            char delimiter = ((d < numDimensions - 1) ? '\t'  : '\n');
            fout << box_muller(mu_i(d), sqrt(sigma_i(d, d))) << delimiter;
        }
    }

    fout.close();
}

/**
 * @brief      Takes input values feature vector x, mean mu, standard deviation 
 * 			   sd, and prior probability P(w_i), and performs the discriminant 
 * 			   function (Case 1).
 *
 * @param[in]  x      The feature vector
 * @param[in]  mu     The mean vector
 * @param[in]  sd     The standard deviation
 * @param[in]  prior  The prior probability P(w_i)
 *
 * @return     The result of processing the discriminant function (1D MatrixXd)
 */
MatrixXd linearDiscFunc_case1(MatrixXd x, MatrixXd mu, float sd, float prior)
{
	MatrixXd mt = mu.transpose();
	MatrixXd w  = (1 / (sd * sd)) * mu;
	MatrixXd wt  = w.transpose();
	MatrixXd w0 = ((-1 / (2 * sd * sd)) * (mt * mu));
	w0(0, 0) += log(prior);

	MatrixXd g_i = (wt * x) + w0;

	return g_i;
}

/**
 * @brief      Takes input values feature vector x, mean mu, covariance matrix 
 * 			   sigma, and prior probability P(w_i), and performs the quadratic discriminant 
 * 			   function (Case 3).
 *
 * @param[in]  x      The feature vector
 * @param[in]  mu     The mean vector
 * @param[in]  sigma  The covariance matrix
 * @param[in]  prior  The prior probability P(w_i)
 *
 * @return     The result of processing the discriminant function (1D MatrixXd)
 */
MatrixXd quadraticDiscFunc_case3(MatrixXd x, MatrixXd mu, MatrixXd sigma, float prior)
{
	MatrixXd xt = x.transpose();
	MatrixXd mt = mu.transpose();
	MatrixXd sigma_inv = sigma.inverse();
	MatrixXd W = -0.5 * sigma_inv;
	MatrixXd w = sigma_inv * mu;
	MatrixXd wt  = w.transpose();
	MatrixXd w0 = (-0.5 * mt * sigma_inv * mu);
	w0(0, 0) -= (0.5 * log(sigma.determinant()));
	w0(0, 0) += log(prior);

	MatrixXd g_i = (xt * W * x) + (wt * x) + w0;

	return g_i;
}

MatrixXd kBound(float beta, MatrixXd mu_1, MatrixXd mu_2, MatrixXd sigma_1, MatrixXd sigma_2){
	float  p_1 = (beta * (1 - beta)) / 2;
	MatrixXd p_2 = mu_1 - mu_2;
	MatrixXd p_3 = ((1-beta) * sigma_1 + beta * sigma_2);

	float p_4 = pow(sigma_1.determinant(), (1 - beta)) * pow(sigma_2.determinant(), beta);
	float p_5 = 0.5 * log(p_3.determinant()/p_4);
	//1x1 matrix return
	MatrixXd part_1 = p_1 * p_2.transpose() * p_3.inverse() * p_2;
	part_1(0,0) += p_5;
	return part_1;
}

//MatrixXd minimumDistanceDiscFunc()












