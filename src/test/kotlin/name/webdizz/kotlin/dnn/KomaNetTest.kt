package name.webdizz.kotlin.dnn

import koma.end
import koma.mat
import kotlin.test.Test
import kotlin.test.assertEquals

class KomaNetTest {

    @Test
    fun shouldTrainNeuralNet() {
        val x = mat[0, 0, 0 end
                0, 0, 1 end
                1, 0, 0 end
                1, 1, 0]
        val y = mat[0, 1, 1, 0].T

        val nn = KomaNet()
        nn.train(x, y)

        assertEquals(0, nn.test(mat[0, 1, 1]))
        assertEquals(1, nn.test(mat[0, 0, 1]))
        assertEquals(1, nn.test(mat[1, 1, 1]))
        assertEquals(0, nn.test(mat[1, 1, 0]))
    }
}