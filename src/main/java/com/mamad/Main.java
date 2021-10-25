package com.mamad;

import com.mamad.activation.ActivationFunction;
import com.mamad.data.MLDataSet;
import com.mamad.network.Network;

public class Main {

    private static final double[][] XOR_INPUT = {
            {1,1},
            {1,0},
            {0,1},
            {0,0}
    };

    private static final double[][] XOR_IDEAL = {
            {0},
            {1},
            {1},
            {0}
    };

    public static void main(String[] args) {

        Network network = new Network(2, 4, 1);
        network.setLearningRate(0.001);
        network.setMomentum(0.5);
        network.setActivationFunction(ActivationFunction.LEAKY_RELU);
        network.runServerAt(8080);

        MLDataSet dataSet = new MLDataSet(XOR_INPUT, XOR_IDEAL);
        network.train(dataSet, 100000);

        network.predict(1,1);
        network.predict(1,0);
        network.predict(0,1);
        network.predict(0,0);

    }
}
