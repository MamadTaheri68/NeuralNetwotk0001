package com.mamad.network;

import com.mamad.activation.ActivationFunction;
import com.mamad.data.MLDataSet;
import org.junit.Assert;
import org.junit.Test;

import static org.junit.Assert.*;

public class NetworkTest {

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

    @Test
    public void xorTestRelu() {
        Network network = new Network(2, 10, 1);
        network.setLearningRate(0.001);
        network.setMomentum(0.6);
        network.setActivationFunction(ActivationFunction.LEAKY_RELU);
        MLDataSet xorSet = new MLDataSet(XOR_INPUT, XOR_IDEAL);

        network.train(xorSet, 100000);

        double[] p1 = network.predict(1,1);
        double[] p2 = network.predict(1,0);
        double[] p3 = network.predict(0,1);
        double[] p4 = network.predict(0,0);

        Assert.assertTrue(p1[0] < 0.1);
        Assert.assertTrue(p2[0] > 0.9);
        Assert.assertTrue(p3[0] > 0.9);
        Assert.assertTrue(p4[0] < 0.1);
    }

    @Test
    public void xorTestSwish() {
        Network network = new Network(2, 10, 1);
        network.setLearningRate(0.001);
        network.setMomentum(0.6);
        network.setActivationFunction(ActivationFunction.SWISH);
        MLDataSet xorSet = new MLDataSet(XOR_INPUT, XOR_IDEAL);

        network.train(xorSet, 100000);

        double[] p1 = network.predict(1,1);
        double[] p2 = network.predict(1,0);
        double[] p3 = network.predict(0,1);
        double[] p4 = network.predict(0,0);

        Assert.assertTrue(p1[0] < 0.1);
        Assert.assertTrue(p2[0] > 0.9);
        Assert.assertTrue(p3[0] > 0.9);
        Assert.assertTrue(p4[0] < 0.1);
    }

    @Test
    public void xorTestTanH() {
        Network network = new Network(2, 10, 1);
        network.setLearningRate(0.001);
        network.setMomentum(0.6);
        network.setActivationFunction(ActivationFunction.LEAKY_RELU);
        MLDataSet xorSet = new MLDataSet(XOR_INPUT, XOR_IDEAL);

        network.train(xorSet, 100000);

        double[] p1 = network.predict(1,1);
        double[] p2 = network.predict(1,0);
        double[] p3 = network.predict(0,1);
        double[] p4 = network.predict(0,0);

        Assert.assertTrue(p1[0] < 0.1);
        Assert.assertTrue(p2[0] > 0.9);
        Assert.assertTrue(p3[0] > 0.9);
        Assert.assertTrue(p4[0] < 0.1);
    }

    @Test
    public void xorTestSigmoid() {
        Network network = new Network(2, 10, 1);
        network.setLearningRate(0.01);
        network.setMomentum(0.6);
        network.setActivationFunction(ActivationFunction.SIGMOID);
        MLDataSet xorSet = new MLDataSet(XOR_INPUT, XOR_IDEAL);

        network.train(xorSet, 100000);

        double[] p1 = network.predict(1,1);
        double[] p2 = network.predict(1,0);
        double[] p3 = network.predict(0,1);
        double[] p4 = network.predict(0,0);

        Assert.assertTrue(p1[0] < 0.1);
        Assert.assertTrue(p2[0] > 0.9);
        Assert.assertTrue(p3[0] > 0.9);
        Assert.assertTrue(p4[0] < 0.1);
    }

}