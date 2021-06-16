# include <iostream>
# include <eigen-3.4/Eigen/Dense>

using Eigen::MatrixXd;

int main() {

    MatrixXd M(2,2);

    M(0,0) = 1;
    M(1,0) = 2;
    M(0,1) = 3;
    M(1,1) = M(0,0) + M(0,1);

    std::cout << M << std::endl;

    return 0;
}