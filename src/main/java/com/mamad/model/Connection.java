package com.mamad.model;

import com.mamad.utils.RandomGenerator;
import lombok.Getter;
import lombok.Setter;

import java.util.UUID;

@Getter
@Setter
public class Connection {

    private UUID connectionId;
    private Neuron from;
    private Neuron to;
    private double synapticWeight;
    private double synapticWeightDelta;

    public Connection(Neuron from, Neuron to) {
        this.connectionId = UUID.randomUUID();
        this.from = from;
        this.to = to;
        this.synapticWeight = RandomGenerator.random(-2, 2);
    }

    public void updateSynapticWeight(double synapticWeight){
        this.synapticWeight += synapticWeight;
    }

}
