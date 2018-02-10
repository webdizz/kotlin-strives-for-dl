package name.webdizz.kotlin.nn

import koma.*
import koma.extensions.*
import koma.matrix.Matrix

class NeuralNet(private val iterationsNum: Int = 1_000_000, private val printEveryIteration: Int = 50_000) {

    var syn0 = 2 * randn(3, 4, 1) - 1
    var syn1 = 2 * randn(4, 1, 1) - 1

    fun train(X: Matrix<Double>, y: Matrix<Double>) {
        for (i in 1..iterationsNum) {
            // start network
            val l0 = X

            val l1 = sigmoid(l0 * syn0)
            val l2 = sigmoid(l1 * syn1)

            // resolve errors to backpropagate
            val l2Error = y - l2
            val l2Delta = l2Error emul sigmoidDx(l2)

            //l1 layer
            val l1Error = l2Delta * syn1.T
            val l1Delta = l1Error * sigmoidDx(l1)

            syn1 += l1.T * l2Delta
            syn0 += l0.T * l1Delta

            if (i % printEveryIteration == 0) {
                println("Error ${mean(abs(l2Error))} ${i}")
            }
        }
    }

    fun test(X: Matrix<Double>): Int {
        val l0 = X
        val l1 = sigmoid(l0 * syn0)
        val l2 = sigmoid(l1 * syn1)

        return if (l2.get(0) > 0.5) 1 else 0
    }

    fun sigmoidDx(x: Matrix<Double>): Matrix<Double> {
        return x emul (1 - x)
    }

    fun sigmoid(x: Matrix<Double>): Matrix<Double> {
        return epow((1 + exp(-x)), -1.0)
    }
}