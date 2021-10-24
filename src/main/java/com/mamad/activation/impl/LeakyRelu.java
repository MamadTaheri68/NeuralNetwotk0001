package main.java.com.mamad.activation.impl;

import main.java.com.mamad.activation.IActivationFunction;

public class LeakyRelu implements IActivationFunction {
    @Override
    public double output(double x) {
        return x >=0 ? x : x * 0.01;
    }

    @Override
    public double outputDerivative(double x) {
        return x >= 0 ? 1 : 0.01;
    }
}