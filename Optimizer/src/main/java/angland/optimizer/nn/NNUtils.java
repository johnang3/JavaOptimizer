package angland.optimizer.nn;

import angland.optimizer.var.MatrixExpression;

public class NNUtils {

  public static <VarKey> MatrixExpression<VarKey> nnLayer(MatrixExpression<VarKey> input,
      int inputSize, int outputSize, VarKey matrixKey, VarKey vectorKey) {
    MatrixExpression<VarKey> matrix = MatrixExpression.variable(matrixKey, outputSize, inputSize);
    MatrixExpression<VarKey> vec = MatrixExpression.variable(vectorKey, outputSize, 1);
    return matrix.times(input).plus(vec);
  }

}
