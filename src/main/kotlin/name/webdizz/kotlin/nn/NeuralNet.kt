package name.webdizz.kotlin.nn

import koma.*
import koma.extensions.*
import koma.matrix.Matrix

class NeuralNet(private val epochs: Int = 10_000, private val printEvery: Int = 1_000) {

    private val seed = setSeed(1)
    private var syn0 = 2 * rand(3, 4) - 1
    private var syn1 = 2 * rand(4, 1) - 1
    private val bias = 0.0

    fun train(x: Matrix<Double>, y: Matrix<Double>) {
        for (i in 1..epochs) {
            // start network training
            val l0 = x

            val l1 = activate(l0 * syn0 + bias)
            val l2 = activate(l1 * syn1 + bias)

            // resolve errors to backpropagate
            val l2Error = y - l2
            val l2Delta = l2Error emul activateDx(l2)

            //l1 layer
            val l1Error = l2Delta * syn1.T
            val l1Delta = l1Error emul activateDx(l1)

            // update weights
            syn1 += l1.T * l2Delta
            syn0 += l0.T * l1Delta

            if (i % printEvery == 0) {
                println("==>Error ${mean(abs(l2Error))}")
            }
        }
    }

    fun activateDx(x: Matrix<Double>): Matrix<Double> {
        return x emul (1 - x)
    }

    fun activate(x: Matrix<Double>): Matrix<Double> {
        return epow((1 + exp(-x)), -1.0)
    }

    fun test(x: Matrix<Double>): Int {
        val l0 = x
        val l1 = activate(l0 * syn0 + bias)
        val l2 = activate(l1 * syn1 + bias)
        return if (abs(l2.get(0)) > 0.3) 1 else 0
    }
}