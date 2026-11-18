#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>

// 1. Helper Functions
double sigmoid(double z) {
    // C++17 clamp to avoid overflow in std::exp
    z = std::clamp(z, -250.0, 250.0);
    return 1.0 / (1.0 + std::exp(-z));
}

std::vector<int> predict(const std::vector<std::vector<double>>& X, const std::vector<double>& weights, double bias) {
    std::vector<int> preds;
    for (const auto& row : X) {
        double linear_model = bias;
        for (size_t i = 0; i < weights.size(); ++i) {
            linear_model += row[i] * weights[i];
        }
        preds.push_back(sigmoid(linear_model) > 0.5 ? 1 : 0);
    }
    return preds;
}

// Minimal CSV state-machine to handle commas inside quotes (e.g., Names)
std::vector<std::string> parse_csv_line(const std::string& line) {
    std::vector<std::string> cols;
    std::string current;
    bool in_quotes = false;
    
    for (char c : line) {
        if (c == '"') {
            in_quotes = !in_quotes;
        } else if (c == ',' && !in_quotes) {
            cols.push_back(current);
            current.clear();
        } else {
            current += c;
        }
    }
    cols.push_back(current);
    return cols;
}

int main() {
    // 2. Data Loading & Preprocessing
    std::ifstream file("titanic.csv");
    if (!file.is_open()) {
        std::cerr << "Error: Could not open titanic.csv. Run wget first.\n";
        return 1;
    }

    std::vector<std::vector<double>> X;
    std::vector<int> y;
    std::string line;
    
    std::getline(file, line); // Skip header

    // Columns expected: 0:PassId, 1:Survived, 2:Pclass, 3:Name, 4:Sex, 5:Age, 6:SibSp, 7:Parch, 8:Ticket, 9:Fare
    while (std::getline(file, line)) {
        auto cols = parse_csv_line(line);
        if (cols.size() < 10) continue;

        int survived = std::stoi(cols[1]);
        double pclass = std::stod(cols[2]);
        double sex = (cols[4] == "female") ? 1.0 : 0.0;
        double age = cols[5].empty() ? 28.0 : std::stod(cols[5]); // 28 is approx median age
        double sibsp = std::stod(cols[6]);
        double parch = std::stod(cols[7]);
        double fare = cols[9].empty() ? 32.2 : std::stod(cols[9]);

        y.push_back(survived);
        X.push_back({pclass, sex, age, sibsp, parch, fare});
    }

    // 3. Train/Test Split (80/20)
    size_t n_samples = X.size();
    std::vector<size_t> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);
    
    std::mt19937 g(42);
    std::shuffle(indices.begin(), indices.end(), g);

    size_t split_idx = n_samples * 0.8;
    std::vector<std::vector<double>> X_train, X_test;
    std::vector<int> y_train, y_test;

    for (size_t i = 0; i < split_idx; ++i) {
        X_train.push_back(X[indices[i]]);
        y_train.push_back(y[indices[i]]);
    }
    for (size_t i = split_idx; i < n_samples; ++i) {
        X_test.push_back(X[indices[i]]);
        y_test.push_back(y[indices[i]]);
    }

    // 4. Normalize Features (Z-score)
    size_t n_features = X_train[0].size();
    std::vector<double> mean(n_features, 0.0), std_dev(n_features, 0.0);
    
    for (const auto& row : X_train) {
        for (size_t j = 0; j < n_features; ++j) mean[j] += row[j];
    }
    for (double& m : mean) m /= X_train.size();

    for (const auto& row : X_train) {
        for (size_t j = 0; j < n_features; ++j) {
            std_dev[j] += (row[j] - mean[j]) * (row[j] - mean[j]);
        }
    }
    for (double& s : std_dev) s = std::sqrt(s / X_train.size());

    auto normalize = [&](std::vector<std::vector<double>>& data) {
        for (auto& row : data) {
            for (size_t j = 0; j < n_features; ++j) {
                if (std_dev[j] > 0) row[j] = (row[j] - mean[j]) / std_dev[j];
            }
        }
    };
    
    normalize(X_train);
    normalize(X_test);

    // 5. Training - Gradient Descent
    std::vector<double> weights(n_features, 0.0);
    double bias = 0.0;
    double learning_rate = 0.1;
    int epochs = 1000;
    size_t m = X_train.size();

    for (int e = 0; e < epochs; ++e) {
        std::vector<double> dw(n_features, 0.0);
        double db = 0.0;

        // Manual matrix multiplication / dot products
        for (size_t i = 0; i < m; ++i) {
            double linear_model = bias;
            for (size_t j = 0; j < n_features; ++j) {
                linear_model += X_train[i][j] * weights[j];
            }
            
            double y_pred = sigmoid(linear_model);
            double error = y_pred - y_train[i];

            for (size_t j = 0; j < n_features; ++j) {
                dw[j] += error * X_train[i][j];
            }
            db += error;
        }

        // Update parameters
        for (size_t j = 0; j < n_features; ++j) {
            weights[j] -= learning_rate * (dw[j] / m);
        }
        bias -= learning_rate * (db / m);
    }

    // 6. Evaluation
    std::vector<int> preds = predict(X_test, weights, bias);
    int correct = 0;
    for (size_t i = 0; i < preds.size(); ++i) {
        if (preds[i] == y_test[i]) correct++;
    }

    double accuracy = static_cast<double>(correct) / y_test.size() * 100.0;
    std::cout << "Accuracy: " << accuracy << "%\n";
    
    std::cout << "Final Weights: ";
    for (double w : weights) std::cout << w << " ";
    std::cout << "\nFinal Bias: " << bias << "\n";

    return 0;
}