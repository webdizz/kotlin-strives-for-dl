package name.webdizz.kotlin.dnn

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import kotlin.test.Test
import kotlin.test.assertTrue

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

        assertTrue(res[0][0] < 0.5)
        assertTrue(res[1][0] > 0.5)
        assertTrue(res[2][0] > 0.5)
        assertTrue(res[3][0] < 0.5)
    }
}