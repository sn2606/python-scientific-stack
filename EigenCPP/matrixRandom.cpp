# include <iostream>
# include <eigen-3.4/Eigen/Dense>

int main() {

    Eigen::MatrixXd M = Eigen::MatrixXd::Random(3,3); // Random constructor and assigning reference
    std::cout << M << std::endl;

    M = M + (Eigen::MatrixXd::Constant(3, 3, 1)) * 10; // It broadcasts like in Python
    std::cout << M << std::endl;

    Eigen::VectorXd V(3);
    V << 1, 2, 3;
    std::cout << V << std::endl;

    Eigen::MatrixXd P = M*V;
    std::cout << P << std::endl;

    return 0;
}