#include <TLorentzVector.h>
#include <TRandom.h>
struct CalResult {
    float Energy;
    float Number;
    vector<vector<float>> Shape;
};

CalResult Calorimeternew(int simuID, TTreeReaderArray<float>& Eng, TTreeReaderArray<int>& simuAssoc, TTreeReaderArray<float>& posX, TTreeReaderArray<float>& posY, TTreeReaderArray<float>& posZ,
                         TTreeReaderArray<unsigned int>& ShPB, TTreeReaderArray<unsigned int>& ShPE, TTreeReaderArray<float>& ShParameters)
{
    float Energy = 0.0, Number = 0.0;
    vector<vector<float>> Shapes;
    
    if (Eng.GetSize() == simuAssoc.GetSize())
    {
        for (int cluster = 0; cluster < Eng.GetSize(); cluster++)
        {
            if (simuID == simuAssoc[cluster])
            {
                Energy += Eng[cluster];
                Number++;

                int start = ShPB[cluster];
                int end   = ShPE[cluster];
                //cout<<"Start: "<<start<<" End: "<<end<<" Size: "<<ShParameters.GetSize()<<endl;

                vector<float> ShapeCluster(11, 0.0f);
                if (ShParameters.GetSize() > 0 && start < (int)ShParameters.GetSize() && ShParameters[start] != 0)
                if (ShParameters[start] != 0)
                {
                    for (int i = start; i < end; i++)
                    {
                        int j = i - start;
                        
                        if (j >= 7) break;
                        ShapeCluster[j] = ShParameters[i];
                    }
                }

                ShapeCluster[7] = Eng[cluster]; 
                ShapeCluster[8] = posX[cluster];
                ShapeCluster[9] = posY[cluster];
                ShapeCluster[10] = posZ[cluster]; 

                Shapes.push_back(ShapeCluster);
            }
        }
    }
    
    return {Energy, Number, Shapes};
}