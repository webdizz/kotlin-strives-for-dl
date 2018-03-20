package name.webdizz.kotlin.dnn

import com.komputation.initialization.heInitialization
import com.komputation.instructions.continuation.activation.Activation
import com.komputation.instructions.continuation.dense.dense
import com.komputation.instructions.entry.input
import com.komputation.instructions.loss.logisticLoss
import com.komputation.matrix.FloatMatrix
import com.komputation.matrix.floatMatrix
import com.komputation.optimization.historical.nesterov
import koma.abs
import koma.extensions.get
import koma.matrix.Matrix
import java.util.*


class KomputationNet(private val epochs: Int = 10_000, private val printEvery: Int = 1_000) {

    private val random = Random(1)

    private val weightInitialization = heInitialization(random)
    private val optimization = nesterov(0.1f, 0.9f)

    private val batchSize = 4
    private val inputDimension = 2
    private val hiddenDimension = 2
    private val outputDimension = 1

    private val network = com.komputation.cpu.network.network(
            batchSize,
            input(inputDimension),
            dense(hiddenDimension, Activation.Sigmoid, weightInitialization, optimization),
            dense(outputDimension, Activation.Sigmoid, weightInitialization, optimization)
    )

    fun train(x: Matrix<Double>, y: Matrix<Double>) {
        val inputs = adoptInput(x)
        val targets = adoptTarget(y)

        network.training(
                inputs = inputs,
                targets = targets,
                numberIterations = epochs,
                loss = logisticLoss(),
                afterEachIteration = printEveryLoss()
        ).run()
    }

    fun test(x: Matrix<Double>): Int {
        val prediction = network.predict(adoptInput(x)[0])
        return if (prediction[0] > 0.3) 1 else 0
    }

    private fun adoptTarget(y: Matrix<Double>): Array<FloatArray> {
        return y.toIterable().map { floatArrayOf(it.toFloat()) }.toTypedArray()
    }

    private fun adoptInput(x: Matrix<Double>): Array<FloatMatrix> {
        return Array(x.numRows()) { rowId ->
            val row = x.getRow(rowId)
            floatMatrix(1, 1, *row.toIterable().map { it.toFloat() }.toFloatArray())
        }
    }

    private fun printEveryLoss() = { iteration: Int, loss: Float ->
        if (iteration % printEvery == 0) {
            println(loss)
        }
    }
}