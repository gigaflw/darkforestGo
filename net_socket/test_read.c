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
    FILE *pFile = fopen("pipe2.txt", "rb");

    Info a;
    fread(&a, sizeof(a), 1, pFile);
    printf("%d\n", a.id);
    printf("%s\n", a.d);
    fclose(pFile);
    return 0;
}
