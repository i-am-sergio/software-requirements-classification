#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm>
using namespace std;

// Función para calcular la media de un vector
double mean(const std::vector<double>& vec) {
    double sum = 0.0;
    for (double v : vec) {
        sum += v;
    }
    return sum / vec.size();
}

// Función para calcular la desviación estándar de un vector
double standard_deviation(const std::vector<double>& vec, double mean_val) {
    double sum = 0.0;
    for (double v : vec) {
        sum += (v - mean_val) * (v - mean_val);
    }
    return std::sqrt(sum / vec.size());
}

// Estandarización de las características (restar la media y dividir por la desviación estándar)
void standardize(std::vector<std::vector<double>>& data) {
    cout<< " --> Estandarización de las características: "<<endl;
    for (size_t j = 0; j < data[0].size(); ++j) {
        std::vector<double> column;
        for (size_t i = 0; i < data.size(); ++i) {
            column.push_back(data[i][j]);
        }

        double mean_val = mean(column);
        double std_dev = standard_deviation(column, mean_val);

        for (size_t i = 0; i < data.size(); ++i) {
            data[i][j] = (data[i][j] - mean_val) / std_dev;
        }
    }
}

// Función para calcular la matriz de covarianza
std::vector<std::vector<double>> covariance_matrix(const std::vector<std::vector<double>>& data) {
    cout<<" --> Matriz de covarianza: "<<endl;
    size_t rows = data.size();
    size_t cols = data[0].size();
    std::vector<std::vector<double>> cov_matrix(cols, std::vector<double>(cols, 0.0));

    for (size_t i = 0; i < cols; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < rows; ++k) {
                sum += data[k][i] * data[k][j];
            }
            cov_matrix[i][j] = sum / (rows - 1);  // Normalizamos
        }
    }

    return cov_matrix;
}

// Función para calcular la transpuesta de una matriz
std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>>& matrix) {
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    std::vector<std::vector<double>> transposed(cols, std::vector<double>(rows));

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            transposed[j][i] = matrix[i][j];
        }
    }

    return transposed;
}

// Función para multiplicar dos matrices
std::vector<std::vector<double>> multiply_matrices(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
    size_t A_rows = A.size();
    size_t A_cols = A[0].size();
    size_t B_rows = B.size();
    size_t B_cols = B[0].size();

    // Imprimir las dimensiones para depuración
    std::cout << "Dimensiones de A: " << A_rows << "x" << A_cols << std::endl;
    std::cout << "Dimensiones de B: " << B_rows << "x" << B_cols << std::endl;

    // Comprobar si las dimensiones son compatibles
    if (A_cols != B_rows) {
        std::cerr << "Error: las dimensiones de las matrices no son compatibles para multiplicación!" << std::endl;
        exit(1);  // Salir si las dimensiones no son compatibles
    }

    std::vector<std::vector<double>> result(A_rows, std::vector<double>(B_cols, 0.0));

    for (size_t i = 0; i < A_rows; ++i) {
        for (size_t j = 0; j < B_cols; ++j) {
            for (size_t k = 0; k < A_cols; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return result;
}


// Función para calcular los vectores propios usando SVD
// Aproximación simple, donde tomamos las primeras componentes principales a partir de la SVD
std::vector<std::vector<double>> eigenvectors(const std::vector<std::vector<double>>& matrix, size_t num_components) {
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();

    // Realizamos una descomposición SVD utilizando una matriz de covarianza.
    std::vector<std::vector<double>> cov_matrix = covariance_matrix(matrix);

    // SVD sería ideal aquí, pero para una implementación sin bibliotecas, necesitamos aproximar
    // Para simplificar, obtenemos los vectores propios directamente de la matriz de covarianza

    // Usamos un "método heurístico" para tomar las primeras `num_components` columnas
    // de la matriz de covarianza para aproximar los vectores principales.
    std::vector<std::vector<double>> eig_vectors(num_components, std::vector<double>(cols, 0.0));

    for (size_t i = 0; i < num_components; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            eig_vectors[i][j] = cov_matrix[i][j];
        }
    }

    return eig_vectors;
}


// Proyección de los datos a las componentes principales
std::vector<std::vector<double>> project_data(const std::vector<std::vector<double>>& data, const std::vector<std::vector<double>>& eigenvectors) {
    cout << " --> Proyección de los datos: "<<endl;
    return multiply_matrices(data, eigenvectors);
}

int main() {
    // Ejemplo de datos (4 ejemplos, 3 características)
    std::vector<std::vector<double>> data = {
        {2.5, 2.4, 3.5},
        {0.5, 0.7, 1.0},
        {2.2, 2.9, 3.2},
        {1.9, 2.2, 2.8}
    };

    // Paso 1: Estandarizar los datos
    standardize(data);

    // Paso 2: Calcular la matriz de covarianza
    auto cov_matrix_result = covariance_matrix(data);

    cout<<" --> Matriz de covarianza: "<<endl;


    // Paso 3: Calcular los vectores propios
    size_t num_components = 2;  // Reducir a 2 dimensiones
    auto eigenvectors_result = eigenvectors(cov_matrix_result, num_components);

    cout << " --> Termina Eigenvectors: "<<endl;
    // Paso 4: Proyectar los datos
    auto projected_data = project_data(data, eigenvectors_result);

    cout << " --> Termina proyección de los datos: "<<endl;

    // Imprimir los resultados
    std::cout << "Datos proyectados (PCA):" << std::endl;
    for (const auto& row : projected_data) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
