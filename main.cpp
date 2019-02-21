#include <iostream>
#include <fstream>
#include <math.h>
#include <cstdlib>
#include <string>
#include <Eigen/Dense>
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

int main()
{
	string outputFile;
	float mean, var;
	double array[NUM_SAMPLES][2];
	string input;

	// number of dimensions for feature vector for each class
	unsigned dim = 2;

	// the filenames for class 1 and class 2
	string filename_1 = "mean1_var1";
	string filename_2 = "mean4_var1";

	// the prior probabilities for class 1 (P(w_1)) and class 2 (P(w_2))
	float pw_1 = 0.5;
	float pw_2 = 0.5;

	// mean matrix for class 1
	VectorXd mu_1(dim);
	mu_1(0) = 1.0;
	mu_1(1) = 1.0;

	// covariance matrix for class 1
	MatrixXd sigma_1(dim, dim);
	sigma_1(0, 0) = 1.0;
	sigma_1(1, 0) = 1.0;
	sigma_1(0, 1) = 1.0;
	sigma_1(1, 1) = 1.0;

	// mean matrix for class 2
	VectorXd mu_2(dim);
	mu_2(0) = 4.0;
	mu_2(1) = 4.0;

	// covariance matrix for class 2
	MatrixXd sigma_2(dim, dim);
	sigma_2(0, 0) = 1.0;
	sigma_2(1, 0) = 1.0;
	sigma_2(0, 1) = 1.0;
	sigma_2(1, 1) = 1.0;

	while (input != "-1")
	{
		cout << endl
		     << "+===============================================+\n"
			 << "|Select  1 to generate new datapoints for part 1|\n"
		     << "|Select  2 to run data on existing data         |\n"
		     << "|Select -1 to exit                              |\n"
		     << "+===============================================+\n"
		     << endl
		     << "Choice: ";

		cin >> input;

		cout << endl;

		if (input == "1")
		{
			srand(SEED);

			// float meanTemp = 1.0;
			// float varTemp  = 1.0;

			// generatePairs(meanTemp, varTemp, array);
			
			//Generate mean4_var1
			// meanTemp = 4.0;
			// generatePairs(meanTemp, varTemp, array);
			
			genSamples(mu_1, sigma_1, dim, filename_1);
			genSamples(mu_2, sigma_2, dim, filename_2);
		}
		else if (input == "2")
		{
			//read from data files
			ifstream fin_G1;
			fin_G1.open(filename_1.c_str());
			ifstream fin_G2;
			fin_G2.open(filename_2.c_str());

			VectorXd xVector(dim, 1);
			float x, y;

			// keep track of how many are classified to 
			// dataset G1 (mean=1,var=1) vs dataset G2 (mean=4,var=1)
			int classifiedAs_i = 0;
			int classifiedAs_j = 0;

			cout << "Running first dataset (" << filename_1 << "):\n\n";

			while (!fin_G1.eof())
			{
				fin_G1 >> x >> y;
				xVector(0,0) = x;
				xVector(1,0) = y;

				//g1Value & g2Value returns a 1-D array
				MatrixXd g1Value = disriminantfunction_Case1_G1(xVector, mu_1, 1.0, pw_1);
				MatrixXd g2Value = disriminantfunction_Case1_G1(xVector, mu_2, 1.0, pw_2);

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

			cout << "Results: G(x) >= 0 (Decide x [Correctly identified]): " 
				 << classifiedAs_i 
				 << ". G(x) < 0 (Decide y [Incorrectly identified]): " 
				 << classifiedAs_j 
				 << ".\n\n";

			// keep track of how many are classified to 
			// dataset G1 (mean=1,var=1) vs dataset G2 (mean=4,var=1)
			classifiedAs_i = 0;
			classifiedAs_j = 0;

			cout << "Running second dataset (" << filename_2 << "):\n\n";
			
			while (!fin_G2.eof())
			{
				fin_G2 >> x >> y;
				xVector(0, 0) = x;
				xVector(1, 0) = y;

				//g1Value & g2Value returns a 1-D array
				MatrixXd g1Value = disriminantfunction_Case1_G1(xVector, mu_1, 1.0, pw_1);
				MatrixXd g2Value = disriminantfunction_Case1_G1(xVector, mu_2, 1.0, pw_2);

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

			cout << "Results: G(x) >= 0 (Decide x [Incorrectly identified]): " 
				 << classifiedAs_i 
				 << ". G(x) < 0 (Decide y [Correctly identified]): " 
				 << classifiedAs_j 
				 << ".\n";
		}
		else if (input != "-1")
		{
			cout << "\"" << input << "\" is not a valid command" << endl;
		}
	}	
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

void generatePairs(float mean, float variance, double valuePair[][2])
{
	//save pairs to file
    ostringstream str1; 
    str1 << mean;
    ostringstream str2; 
    str2 << variance;

	ofstream fout;
	string fileName = "mean"+str1.str()+"_var"+str2.str();

	cout << "Generating data for " << fileName << "." << endl;

	//generate pairs
	for (int i = 0; i < NUM_SAMPLES; ++i)
	{
		//Sampling x & y values
		valuePair[i][0] = box_muller(mean, sqrt(variance));
		valuePair[i][1] = box_muller(mean, sqrt(variance));
		//cout << valuePair[i][0] << '\t' << valuePair[i][1] << endl;
	}

	fout.open(fileName.c_str());

	for (int i = 0; i < NUM_SAMPLES; ++i)
	{
		fout << valuePair[i][0] << '\t' << valuePair[i][1] << endl;
	}
	fout.close();
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

void useBayesianClassifier(string dataFile)
{

}

/**
 * @brief      Takes input values feature vector x, mean mu, standard deviation 
 * 			   sd, and prior probability P(w_i), and performs the discriminant 
 * 			   function.
 *
 * @param[in]  x      The feature vector
 * @param[in]  mu     The mean vector
 * @param[in]  sd     The standard deviation
 * @param[in]  prior  The prior probability P(w_i)
 *
 * @return     The result of processing the discriminant function (1D MatrixXd)
 */
MatrixXd disriminantfunction_Case1_G1(MatrixXd x, MatrixXd mu, float sd, float prior)
{
	MatrixXd xt = x.transpose();
	MatrixXd mt = mu.transpose();
/*
	MatrixXd w_i = (1/sd*sd)*mu;
	cout << (1/sd*sd)*mu << endl;
	MatrixXd w_i0 = (-1/2*sd*sd)*mt*mu;
	float endingPartOfEquation = log(.5); //assumes P(w_i) == P(w_j) therefore -> .5

	MatrixXd g_i_part1 = w_i.transpose()*x;
*/
	MatrixXd g_i = (-1 / (2 * sd * sd)) * ((xt * x) - (2 * mt * x) + (mt * mu));
	g_i(0, 0) += log(prior);
	//add + ln(P_wi)

	/*cout << "xt*x: " << xt*x << endl;
	cout << "2*mt*x + mt*mu: " << 2*mt*x + mt*mu << endl;

	cout << "(-1/2*sd*sd): " <<(-1/(2*sd*sd)) << endl;
	cout << "(xt*x - 2*mt*x + mt*mu): " << (xt*x - 2*mt*x + mt*mu) << endl;
	cout << g_i << endl;*/
	return g_i;
	//float endingPartOfEquation = log(.5); //assumes P(w_i) == P(w_j) therefore -> .5

}

//get equation from g1=g2
//should get y=-x 