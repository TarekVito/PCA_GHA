using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace NN_Project
{
    class PCA
    {
        double ETA, maxIters;
        int neuronsNum;
        List<List<double>> weights, weightsTmp;
        List<double> output;

        public PCA(double _ETA, double _maxIters, int _neuronsNum,int featureSize)
        {
            ETA = _ETA;
            maxIters = _maxIters;
            neuronsNum = _neuronsNum;
            output = new List<double>();
            weightsTmp = new List<List<double>>();

            // Initializing Random weights for the first iteration..
            weights = new List<List<double>>();
            Random rand = new Random();
            for (int i = 0; i < neuronsNum; i++)
            {
                weights.Add(new List<double>());
                for (int j = 0; j < featureSize; j++)
                    weights[i].Add(rand.NextDouble());
            }
        }
        public void startTraining(List<List<double>> trainingData)
        {
            int t, epoch;
            for (epoch = 0; epoch < maxIters; ++epoch)
            {
                for (t = 0; t < trainingData.Count; ++t)
                {
                    weightsTmp = new List<List<double>>();
                    for (int i = 0; i < weights.Count; ++i)
                        weightsTmp.Add(new List<double>(weights[i]));
                    
                    output = reduceSampleFeatures(trainingData[t]);
                    updateWeights(trainingData[t]);
                    
                    if (equalWeight(weights,weightsTmp))     // if old weight == new weight -> stop training
                        break;
                }
                if (t < trainingData.Count)
                    break;
            }
        }
        
        public List<double> reduceSampleFeatures(List<double> sample)
        {
            List<double> output = new List<double>();
            for(int i=0;i<neuronsNum;++i)
                output.Add(0);

            Parallel.For(0, neuronsNum, j =>
            {
                for (int i = 0; i < sample.Count; i++) //900
                {
                    output[j] += sample[i] * weights[j][i];
                }
            });
            return output;
        }
        private void updateWeights(List<double> trainingData)
        {
            Parallel.For(0, weights.Count, j =>
            //for (int j = 0; j < weights.Count; j++) //100
            {
                //double sumTerm = getSum(j);
                for (int i = 0; i < weights[j].Count; i++) //900
                {
                    //update weights..
                    double sumTerm = getSum2(j,i);
                    weights[j][i] += ETA * output[j] * (trainingData[i] - sumTerm);
                }
            });
        }
        private bool equalWeight(List<List<double>> w1,List<List<double>> w2)
        {
            for (int i = 0; i < w1.Count; ++i)
                for (int j = 0; j < w1[i].Count; ++j)
                    if (w1[i][j] != w2[i][j])
                        return false;
            return true;
        }
        private double getSum(int jj)
        {
            double sum = 0;
            for (int j = 0; j <= jj; j++)
            {
                for (int i = 0; i < weightsTmp[j].Count; i++) //900
                {
                    sum += output[j] * weightsTmp[j][i];
                }
            }
            return sum;
        }
        private double getSum2(int jj,int ii)
        {
            double sum = 0;
            for (int j = 0; j <= jj; j++)
                sum += output[j] * weightsTmp[j][ii];
            return sum;
        }
    }
}
