//
// Created by HgS_1217_ on 2018/1/8.
//

#include <stdlib.h>
#include <stdio.h>

typedef struct {
    int id;
    char *d;
} Info;

int main() {
    FILE *pFile;
    Info a;
    a.id = 1217;
    a.d = "test";
    pFile = fopen("pipe.txt", "wb");

    fwrite(&a, sizeof(a), 1, pFile);
    fclose(pFile);
    return 0;
}
