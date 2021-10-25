package com.mamad.network;

import com.mamad.activation.ActivationFunction;
import com.mamad.data.MLData;
import com.mamad.data.MLDataSet;
import com.mamad.model.Neuron;
import com.mamad.server.MultiLayerNetworkView;
import fi.iki.elonen.NanoHTTPD;
import main.java.com.mamad.activation.IActivationFunction;
import main.java.com.mamad.activation.impl.LeakyRelu;
import main.java.com.mamad.activation.impl.Sigmoid;
import main.java.com.mamad.activation.impl.Swish;
import main.java.com.mamad.activation.impl.TanH;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class Network {

    private static final Logger logger = LogManager.getLogger(Network.class);

    private final int inputSize;
    private final int hiddenSize;
    private final int outputSize;

    private final List<Neuron> inputLayer;
    private final List<Neuron> hiddenLayer;
    private final List<Neuron> outputLayer;

    private double learningRate = 0.01;
    private double momentum = 0.5;
    private IActivationFunction activationFunction = new Sigmoid(); // default activation function

    public Network(int inputSize, int hiddenSize, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.inputLayer = new ArrayList<>();
        this.hiddenLayer = new ArrayList<>();
        this.outputLayer = new ArrayList<>();
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public void setMomentum(double momentum) {
        this.momentum = momentum;
    }

    public void setActivationFunction(ActivationFunction activationFunction) {
        switch (activationFunction) {
            case LEAKY_RELU:
                this.activationFunction = new LeakyRelu();
                break;
            case TANH:
                this.activationFunction = new TanH();
                break;
            case SIGMOID:
                this.activationFunction = new Sigmoid();
                break;
            case SWISH:
                this.activationFunction = new Swish();
                break;
        }
    }

    private void init() {
        for (int i = 0; i < inputSize; i++) {
            this.inputLayer.add(new Neuron());
        }
        for (int i = 0; i < hiddenSize; i++) {
            this.hiddenLayer.add(new Neuron(this.inputLayer, activationFunction));
        }
        for (int i = 0; i < outputSize; i++) {
            this.outputLayer.add(new Neuron(this.hiddenLayer, activationFunction));
        }
        logger.info("Network Initialized.");
    }

    public void train(MLDataSet set, int epoch){
        this.init();
        logger.info("Training Started");
        for (int i = 0; i < epoch; i++) {
            Collections.shuffle(set.getData());
            for (MLData datum : set.getData()) {
                forward(datum.getInputs());
                backward(datum.getTargets());
            }
        }
        logger.info("Trainig Finished.");
    }

    private void backward(double[] targets) {
        int i=0;
        for (Neuron neuron : outputLayer) {
            neuron.calculateGradient(targets[i++]);
        }
        for (Neuron neuron : hiddenLayer) {
            neuron.calculateGradient();
        }
        for (Neuron neuron : hiddenLayer) {
            neuron.updateConnections(learningRate, momentum);
        }
        for (Neuron neuron : outputLayer) {
            neuron.updateConnections(learningRate, momentum);
        }
    }

    //forward propagation
    private void forward(double[] inputs) {
        int i = 0;
        for (Neuron neuron : inputLayer) {
            neuron.setOutput(inputs[i++]);
        }
        for (Neuron neuron : hiddenLayer) {
            neuron.calculateOutput();
        }
        for (Neuron neuron : outputLayer) {
            neuron.calculateOutput();
        }
    }

    public double[] predict(double... inputs) {
        forward(inputs);
        double[] output = new double[outputLayer.size()];
        for (int i = 0; i < output.length; i++) {
            output[i] = outputLayer.get(i).getOutput();
        }
        logger.info("Input : "+ Arrays.toString(inputs) + " Predicted : " + Arrays.toString(output));
        return output;
    }

    public void runServerAt(int port) {

        double[] layers = new double[3];
        layers[0] = inputSize;
        layers[1] = hiddenSize;
        layers[2] = outputSize;

        MultiLayerNetworkView.DATA_NETWORK = Arrays.toString(layers);
        MultiLayerNetworkView networkView = new MultiLayerNetworkView(port);
        try {
            networkView.start(NanoHTTPD.SOCKET_READ_TIMEOUT,false);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
