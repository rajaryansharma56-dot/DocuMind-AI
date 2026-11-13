
#include<iostream>
#include<string>

using namespace std;

int main(){

    int n;
    cout<<"enter the number of students:";
    cin>>n;

    int student_count_above_75=0;
    int student_count_below_75=0;

    int highest_attendance=-1;
    string highest_student;

    for(int i=0; i<n; i++){

        cout<<"enter student name "<<i+1<<":";
        string name;
        cin>>name;

        int attendance_percentage;
        cout<<"enter attendance percentage of student "<<i+1<<":";
        cin>>attendance_percentage;

        if(attendance_percentage<0 || attendance_percentage>100){
            cout<<"invalid attendance"<<endl;
            i--;
            continue;
        }

        if(attendance_percentage>=75){
            student_count_above_75++;
        }
        else{
            student_count_below_75++;
        }

        if(attendance_percentage>highest_attendance){
            highest_attendance=attendance_percentage;
            highest_student=name;
        }
    }

    cout<<"\nTotal Students: "<<n<<endl;
    cout<<"Eligible Students: "<<student_count_above_75<<endl;
    cout<<"Attendance Shortage: "<<student_count_below_75<<endl;

    cout<<"Highest Attendance: "<<highest_student
        <<" - "<<highest_attendance<<"%"<<endl;

    cout<<"Final Report: Generated"<<endl;

    return 0;
}
