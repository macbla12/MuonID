#include <TLorentzVector.h>
#include <TRandom.h>
struct CalResult {
    double Energy;
    double Number;
    vector<float> Shape;
};

CalResult Calorimeternew(int simuID,TTreeReaderArray<float>& Eng, TTreeReaderArray<int>&simuAssoc, TTreeReaderArray<unsigned int>&ShPB, TTreeReaderArray<unsigned int>& ShPE, TTreeReaderArray<float>& ShParameters)
{
    double Energy=0.0,Number=0.0;
    vector<float> Shape;
    Shape.resize(7,0);
    if(Eng.GetSize()==simuAssoc.GetSize())
    {
        double maxEnergy=0;
        for(int cluster=0;cluster<Eng.GetSize();cluster++)
        {
            //cout<<"Before   "<<simuID<<"     "<<simuAssoc[cluster]<<endl;
            if(simuID==simuAssoc[cluster])
            {
                //cout<<"After"<<simuID<<"     "<<simuAssoc[cluster]<<endl;


                if(Eng[cluster]>maxEnergy) maxEnergy=Eng[cluster];

                Energy+=Eng[cluster];
                Number++;

                int start = ShPB[cluster];
                int end   = ShPE[cluster];
        
                for(int i = start; i < end; i++) {
                    if(Eng[cluster]<maxEnergy) break;
                    if(ShParameters[0]==0) break;
                    int j=i-start;
                    Shape[j] = ShParameters[i]; 

                }

                
            } 
        } 
    }

    return {Energy,Number,Shape};
}