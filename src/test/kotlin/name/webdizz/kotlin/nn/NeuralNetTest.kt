package name.webdizz.kotlin.nn

import koma.*
import org.junit.Test
import kotlin.test.assertEquals

class NeuralNetTest {

    @Test
    fun shouldUseNeuralNet() {
        val X = mat[0, 0, 1 end
                0, 1, 1 end
                1, 0, 1 end
                1, 1, 1]
        val y = mat[0, 1, 1, 0].T

        val nn = NeuralNet()
        nn.train(X, y)

        assertEquals(1, nn.test(mat[0, 1, 1]))
        assertEquals(0, nn.test(mat[0, 0, 1]))
    }
}

