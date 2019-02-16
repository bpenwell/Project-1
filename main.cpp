#include <iostream>
#include <fstream>
#include <math.h>
#include <cstdlib>
#include <string>

//sdlkjgj

using namespace std;

//Created by Ben Penwell and Adam <last name>
//Pattern Recognition, Project 1
//Feb. 12, 2019
float ranf();
float box_muller(float m, float s);
void generatePairs(float mean, float variance, double array[][2]);

int main(){
	srand (time(NULL));

	string outputFile;
	float mean, var;
	double array[100000][2];

	int input;
	cout << "Select 1 to generate datapoints: " << endl;
	cin >> input;
	if(input==1){
		generatePairs(1,1,array);
		//plotdata x,y;
		for (int i = 0; i < 100000; ++i)
		{
			//addMark(x,y,array[i][0],array[i][1]);
		}
		//plot(x,y);
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

void generatePairs(float mean, float variance, double valuePair[][2]){
	//generate pairs
	for (int i = 0; i < 100000; ++i)
	{
		//Sampling x & y values
		valuePair[i][0] = box_muller(mean,variance);
		valuePair[i][1] = box_muller(mean,variance);
		//cout << valuePair[i][0] << '\t' << valuePair[i][1] << endl;
	}

	//save pairs to file
	ofstream fout;
	string fileName = "mean1_var1";
	fout.open(fileName.c_str());

	for (int i = 0; i < 100000; ++i)
	{
		fout << valuePair[i][0] << '\t' << valuePair[i][1] << endl;
	}
	fout.close();
}