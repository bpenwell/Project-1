#include <iostream>
#include <fstream>
#include <math.h>
#include <cstdlib>
#include <string>
#include <Eigen/Dense>
#include <math.h>

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
void useBayesianClassifier(string dataFile);
MatrixXd disriminantfunction_Case1_G1(MatrixXd x_Matrix, 
									  MatrixXd mean, 
									  float variance, 
									  float probability);

int main()
{
	string outputFile;
	float mean, var;
	double array[NUM_SAMPLES][2];
	string input;

	while (input != "-1")
	{
		cout << "Select 1 to generate new datapoints for part 1"
		     << "select 2 to run data on existing data, -1 to exit: ";

		cin >> input;

		if (input == "1")
		{
			srand(SEED);
			
			cout << "Generating data for mean1_var1." << endl;

			float meanTemp = 1.0;
			float varTemp  = 1.0;
			generatePairs(meanTemp, varTemp, array);
			
			//Generate mean4_var1
			meanTemp = 4.0;
			generatePairs(meanTemp, varTemp, array);
		}
		else if (input == "2")
		{
			//Set mean matrix G1
			MatrixXd meanMatrix_G1(2,1);
			meanMatrix_G1(0,0)=1.0;
			meanMatrix_G1(1,0)=1.0;
			//Set mean matrix G2
			MatrixXd meanMatrix_G2(2,1);
			meanMatrix_G2(0,0)=4.0;
			meanMatrix_G2(1,0)=4.0;

			//read from data files
			ifstream fin_G1;
			fin_G1.open("mean1_var1");
			ifstream fin_G2;
			fin_G2.open("mean4_var1");

			MatrixXd xVector(2,1);
			float x,y;

			// keep track of how many are classified to 
			// dataset G1 (mean=1,var=1) vs dataset G2 (mean=4,var=1)
			int classifiedAs_i = 0;
			int classifiedAs_j = 0;

			cout << "Running first dataset (mean1_var1):\n\n";
			while (!fin_G1.eof())
			{
				fin_G1 >> x >> y;
				xVector(0,0) = x;
				xVector(1,0) = y;

				//g1Value & g2Value returns a 1-D array
				MatrixXd g1Value = disriminantfunction_Case1_G1(xVector, meanMatrix_G1, 1.0, 0.2);
				MatrixXd g2Value = disriminantfunction_Case1_G1(xVector, meanMatrix_G2, 1.0, 0.8);

				float temp = g1Value(0,0) - g2Value(0,0);

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

			cout << "\nRunning second dataset (mean4_var1):\n\n";
			
			while (!fin_G2.eof())
			{
				fin_G2 >> x >> y;
				xVector(0,0)=x;
				xVector(1,0)=y;

				//g1Value & g2Value returns a 1-D array
				MatrixXd g1Value = disriminantfunction_Case1_G1(xVector, meanMatrix_G1, 1.0, 0.2);
				MatrixXd g2Value = disriminantfunction_Case1_G1(xVector, meanMatrix_G2, 1.0, 0.8);

				float temp = g1Value(0,0)-g2Value(0,0);

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

double ranf(double m){
	return (m*rand()/(double)RAND_MAX);
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
	//generate pairs
	for (int i = 0; i < NUM_SAMPLES; ++i)
	{
		//Sampling x & y values
		valuePair[i][0] = box_muller(mean, sqrt(variance));
		valuePair[i][1] = box_muller(mean, sqrt(variance));
		//cout << valuePair[i][0] << '\t' << valuePair[i][1] << endl;
	}

	//save pairs to file
    ostringstream str1; 
    str1 << mean;
    ostringstream str2; 
    str2 << variance;

	ofstream fout;
	string fileName = "mean"+str1.str()+"_var"+str2.str();
	fout.open(fileName.c_str());

	for (int i = 0; i < NUM_SAMPLES; ++i)
	{
		fout << valuePair[i][0] << '\t' << valuePair[i][1] << endl;
	}
	fout.close();
}

void useBayesianClassifier(string dataFile)
{

}

//passes input values x_Matrix, mean matrix, variance matrix, and probability P(wi)
MatrixXd disriminantfunction_Case1_G1(MatrixXd x_Matrix, 
									  MatrixXd mean, 
									  float variance, 
									  float probability)
{
/*
	MatrixXd w_i = (1/variance*variance)*mean;
	cout << (1/variance*variance)*mean << endl;
	MatrixXd w_i0 = (-1/2*variance*variance)*mean.transpose()*mean;
	float endingPartOfEquation = log(.5); //assumes P(w_i) == P(w_j) therefore -> .5

	MatrixXd g_i_part1 = w_i.transpose()*x_Matrix;
*/
	MatrixXd g_i = (-1/(2*variance))*(x_Matrix.transpose()*x_Matrix - 2*mean.transpose()*x_Matrix + mean.transpose()*mean);
	g_i(0,0) += log(probability);
	//add + ln(P_wi)

	/*cout << "x_Matrix.transpose()*x_Matrix: " << x_Matrix.transpose()*x_Matrix << endl;
	cout << "2*mean.transpose()*x_Matrix + mean.transpose()*mean: " << 2*mean.transpose()*x_Matrix + mean.transpose()*mean << endl;

	cout << "(-1/2*variance*variance): " <<(-1/(2*variance*variance)) << endl;
	cout << "(x_Matrix.transpose()*x_Matrix - 2*mean.transpose()*x_Matrix + mean.transpose()*mean): " << (x_Matrix.transpose()*x_Matrix - 2*mean.transpose()*x_Matrix + mean.transpose()*mean) << endl;
	cout << g_i << endl;*/
	return g_i;
	//float endingPartOfEquation = log(.5); //assumes P(w_i) == P(w_j) therefore -> .5

}

//get equation from g1=g2
//should get y=-x 