package com.mamad.utils;

public class RandomGenerator {
    public static double random(int min, int max) {
        return min + (max - min) * Math.random();
    }
}
