package name.webdizz.kotlin.dnn

import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.math.exp
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.rand
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import org.jetbrains.kotlinx.multik.ndarray.operations.*

class NeuralNet(private val epochs: Int = 10_000, private val printEvery: Int = 1_000) {

    private val syn0 = 2.0 * mk.rand(3, 4) - 1.0
    private val syn1 = 2.0 * mk.rand(4, 1) - 1.0

    fun train(x: NDArray<Double, D2>, y: NDArray<Double, D2>): NDArray<Double, D2> {
        var out: NDArray<Double, D2> = mk.zeros(4, 1)
        for (i in 1..epochs) {
            val l0 = x
            val l1 = activate(l0.dot(syn0))
            val l2 = activate(l1.dot(syn1))

            // what is a difference between expected and predicted values
            val l2e = y - l2

            // calculate delta to scale our weights according to activated gradient
            val l2d = l2e * activateDx(l2)

            // find out how weights of hidden layer were responsible in error of output layer
            val l1e = l2d dot syn1.reshape(1, 4)

            // find out a gradient/slope of a function we're trying to learn for each new output
            // taking into account responsibility of the hidden layer and scale it by the responsibility of hidden layer in final error
            val l1d = l1e * activateDx(l1)

            syn0 += l0.transpose() dot l1d
            syn1 += l1.transpose() dot l2d
            if (i == epochs) {
                out = l2
            }
            if (i % printEvery == 0) {
                println("Error: ${l2e.transpose()}")
            }
        }
        return out
    }

    private fun activate(x: NDArray<Double, D2>): NDArray<Double, D2> {
        return 1.0 / (1.0 + (-x).exp())
    }

    private fun activateDx(x: NDArray<Double, D2>): NDArray<Double, D2> {
        return x * (1.0 - x)
    }
}