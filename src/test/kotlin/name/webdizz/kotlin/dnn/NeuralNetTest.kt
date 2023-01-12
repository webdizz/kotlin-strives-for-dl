package name.webdizz.kotlin.dnn

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import kotlin.test.Test

class NeuralNetTest {

    @Test
    fun shouldCalculate() {
        val x = mk.ndarray(mk[
                mk[0.0, 0.0, 0.0],
                mk[0.0, 0.0, 1.0],
                mk[1.0, 0.0, 0.0],
                mk[1.0, 1.0, 0.0],
        ])

        val y = mk.ndarray(mk[0.0, 1.0, 1.0, 0.0]).reshape(4, 1)

        val net = NeuralNet(10_000, 1000);
        val res = net.train(x, y)

        println(res)
    }
}