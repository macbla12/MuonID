#include <TLorentzVector.h>
#include <TRandom.h>
struct CalResult {
    double ECalEnergy;
    double ECalNumber;
    double HCalEnergy;
    double HCalNumber;
};

CalResult Calorimeters(TLorentzVector particle, int simuID,
            TTreeReaderArray<float>& EcalBarrelEng, TTreeReaderArray<float>& EcalEndcapPEng, TTreeReaderArray<float>& EcalEndcapNEng, 
            TTreeReaderArray<float>& HcalBarrelEng, TTreeReaderArray<float>& HcalEndcapPEng, TTreeReaderArray<float>& LFHcalEng, TTreeReaderArray<float>& HcalEndcapNEng,
            TTreeReaderArray<float>& B0Eng,TTreeReaderArray<float>& EcalBarrelImagingEng,TTreeReaderArray<float>& EcalBarrelScFiEng,
            TTreeReaderArray<unsigned int>& simuAssocEcalBarrel, TTreeReaderArray<unsigned int>& simuAssocEcalEndcapP, TTreeReaderArray<unsigned int>& simuAssocEcalEndcapN,
            TTreeReaderArray<unsigned int>& simuAssocHcalBarrel, TTreeReaderArray<unsigned int>& simuAssocHcalEndcapP, TTreeReaderArray<unsigned int>& simuAssocLFHcal,
            TTreeReaderArray<unsigned int>& simuAssocHcalEndcapN, TTreeReaderArray<unsigned int>&simuAssocB0, TTreeReaderArray<unsigned int>&simuAssocEcalBarrelImaging,
            TTreeReaderArray<unsigned int>&simuAssocEcalBarrelScFi, TTreeReaderArray<unsigned int>&B0ShPB, TTreeReaderArray<unsigned int>& B0ShPE, TTreeReaderArray<float>& B0ShParameters)
{
    double ECalEnergy=0.0,ECalNumber=0.0,HCalEnergy=0.0,HCalNumber=0.0;
    vector<vector<float>> ShapeParameters;
    if(B0Eng.GetSize()==simuAssocB0.GetSize())
    {
        for(int cluster=0;cluster<B0Eng.GetSize();cluster++)
        {
            
            if(simuID==simuAssocB0[cluster])
            {
                
                ECalEnergy+=B0Eng[cluster];
                ECalNumber++;

                int start = B0ShPB[cluster];
                int end   = B0ShPE[cluster];
                vector<float> temp(7);
                cout<<"Start"<<endl;        
                for(int i = start; i < end; i++) {
                    if(B0ShParameters[0]==0) break;

                    int j=i-start;
                    temp[j] = B0ShParameters[i]; 

                    cout<<temp[j]<<endl;
                }

                ShapeParameters.push_back(temp);
            } 
        } 
    }

    if(EcalBarrelImagingEng.GetSize()==simuAssocEcalBarrelImaging.GetSize())
    {
        for(int cluster=0;cluster<EcalBarrelImagingEng.GetSize();cluster++)
        {
            
            if(simuID==simuAssocEcalBarrelImaging[cluster])
            {
                
                ECalEnergy+=EcalBarrelImagingEng[cluster];
                ECalNumber++;
            } 
        } 
    }

    if(EcalBarrelScFiEng.GetSize()==simuAssocEcalBarrelScFi.GetSize())
    {
        for(int cluster=0;cluster<EcalBarrelScFiEng.GetSize();cluster++)
        {
            
            if(simuID==simuAssocEcalBarrelScFi[cluster])
            {
                
                ECalEnergy+=EcalBarrelScFiEng[cluster];
                ECalNumber++;
            } 
        } 
    }
    if(EcalBarrelEng.GetSize()==simuAssocEcalBarrel.GetSize())
    {
        for(int cluster=0;cluster<EcalBarrelEng.GetSize();cluster++)
        {
            
            if(simuID==simuAssocEcalBarrel[cluster])
            {
                
                ECalEnergy+=EcalBarrelEng[cluster];
                ECalNumber++;
            } 
        } 
    }
    if(EcalEndcapPEng.GetSize()==simuAssocEcalEndcapP.GetSize())
    {
        for(int cluster=0;cluster<EcalEndcapPEng.GetSize();cluster++)
            if(simuID==simuAssocEcalEndcapP[cluster])  
            {
                ECalEnergy+=EcalEndcapPEng[cluster];
                ECalNumber++;
            }  
    }      
    if(EcalEndcapNEng.GetSize()==simuAssocEcalEndcapN.GetSize())
    {
        for(int cluster=0;cluster<EcalEndcapNEng.GetSize();cluster++)
            if(simuID==simuAssocEcalEndcapN[cluster])  
            {
                ECalEnergy+=EcalEndcapNEng[cluster];
                ECalNumber++;
            }  
    }


    //Hcal Energy Search
    if(HcalBarrelEng.GetSize()==simuAssocHcalBarrel.GetSize())
    {
        for(int cluster=0;cluster<HcalBarrelEng.GetSize();cluster++)
            if(simuID==simuAssocHcalBarrel[cluster])
            {
                HCalEnergy+=HcalBarrelEng[cluster];
                HCalNumber++;
            }  
    }
    if(HcalEndcapPEng.GetSize()==simuAssocHcalEndcapP.GetSize())
    {
        for(int cluster=0;cluster<HcalEndcapPEng.GetSize();cluster++)           
            if(simuID==simuAssocHcalEndcapP[cluster])  
            {
                HCalEnergy+=HcalEndcapPEng[cluster];
                HCalNumber++;
            }               
    }
    if(HcalEndcapNEng.GetSize()==simuAssocHcalEndcapN.GetSize())
    {
        for(int cluster=0;cluster<HcalEndcapNEng.GetSize();cluster++)
            if(simuID==simuAssocHcalEndcapN[cluster])     
            {
                HCalEnergy+=HcalEndcapNEng[cluster]; 
                HCalNumber++;
            }           
    }
    if(LFHcalEng.GetSize()==simuAssocLFHcal.GetSize())
    {
        for(int cluster=0;cluster<LFHcalEng.GetSize();cluster++)           
            if(simuID==simuAssocLFHcal[cluster])        
            {
                HCalEnergy+=LFHcalEng[cluster];
                HCalNumber++;
            }       
    }
    return {ECalEnergy,ECalNumber,HCalEnergy,HCalNumber};
}