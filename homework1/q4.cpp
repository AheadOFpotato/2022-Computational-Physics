#include <iostream>
#include <cmath>
using namespace std;
int main()
{
    int r = 8000;
    float q2 = 0.5;
    double sum = 0;
    double q = sqrt(q2);
    for (int x = -r; x < r + 1; x++)
    {
        if (x % 500 == 0)
        {
            cout << x << endl;
        }
        int ylim = sqrt(r * r - x * x);
        for (int y = -ylim; y < ylim + 1; y++)
        {
            int zlim = sqrt(r * r - x * x - y * y);
            for (int z = -zlim; z < zlim + 1; z++)
            {
                sum = sum + 1 / (x * x + y * y + z * z - q2);
            }
        }
    }
    double pi = 3.14159265358979;
    double inter = 4 * pi * (r - q * atanh(q / r));
    cout << sum << endl;
    cout << inter << endl;
    cout << (sum - inter) << endl;

    return 0;
}
