#include<iostream>

using namespace std;

int main(){
    int n;
    cout<<"enter number of readings:";
    cin>>n;

    int highest=0;
    int lowest=0;

    int above_50=0;

    for(int i=0; i<n; i++){
        int readings;
        cout<<"enter reading "<<i+1<<":";
        cin>>readings;

        
        if(readings<0){
            cout<<"invalid readings"<<endl;
        }


        else{

        

        if(readings>highest){
            highest=readings;
        }

        if(readings<lowest){
            lowest=readings;
        }

        if(readings>50){
            above_50++;
        }

        if(readings>20){
            cout<<"reading is above 20"<<endl;
        }

        else{
            cout<<"NORMAL"<<endl;
        }

        
    }

    cout<<"highest reading is:"<<highest<<endl;
    cout<<"lowest reading is:"<<lowest<<endl;
    cout<<"number of readings above 50 is:"<<above_50<<endl;
    cout<<"number of readings above 20 is:"<<n-above_50<<endl;
    cout<<"number of readings below 20 is:"<<n-above_50<<endl;
    cout<<"number of readings below 50 is:"<<n-above_50<<endl;

    return 0;
}