#include <stdio.h>
#include <string>
#include "DenseTrack.h"

using namespace std;

int main(int argc, char** argv) {
    
    char start_frame[20];
    char end_frame[20];
    
    const int fake_argc = 11;
    char *fake_argv[fake_argc];
    fake_argv[0] = argv[1];
    fake_argv[1] = (char*)"-S";
    fake_argv[2] = start_frame;
    fake_argv[3] = (char*)"-E";
    fake_argv[4] = end_frame;
    fake_argv[5] = (char*)"-L";
    fake_argv[6] = (char*)"15";
    fake_argv[7] = (char*)"-W";
    fake_argv[8] = (char*)"5";
    fake_argv[9] = (char*)"-A";
    fake_argv[10] = (char*)"1";
    
    int state = 0;
    int current_frame = 0;
    int current_index = 1;
    char index_str[5];
    while (true) {
        sprintf(index_str, "%04d", current_index);
        string out_filename = string(argv[2]) + "_" + index_str + ".txt";
        FILE* out = freopen(out_filename.c_str(), "w", stdout);
        if (!out) {
            printf("Cannot open file.\n");
            break;
        }
        
        sprintf(start_frame, "%d", current_frame);
        sprintf(end_frame, "%d", current_frame + 15);
        state = DenseTrack(fake_argc, fake_argv);
        
        fclose(out);
        
        if (state) {
            remove(out_filename.c_str());
            break;
        }
        
        current_frame += 15;
        current_index += 1;
    };
    
    return 0;
    
}