package angland.optimizer.var;

import static org.junit.Assert.assertEquals;

import java.util.HashMap;
import java.util.Map;

import org.junit.Ignore;
import org.junit.Test;

public class MatrixTest {

  private static final double TOLERANCE = 10e-3;

  /**
   * Verify a simple case of matrix multiplication. <br/>
   * <br/>
   * | 1,2,3 | |1,2| | 22, 28 | <br/>
   * | 4,5,6 | x |3,4| = | 49, 64 | <br/>
   * |5,6|
   */
  @Test
  public void testMatrixMultiply() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    MatrixExpression<String> leftMatrix = MatrixExpression.variable("left", 2, 3);
    context.put(IndexedKey.matrixKey("left", 0, 0), 1.0);
    context.put(IndexedKey.matrixKey("left", 0, 1), 2.0);
    context.put(IndexedKey.matrixKey("left", 0, 2), 3.0);
    context.put(IndexedKey.matrixKey("left", 1, 0), 4.0);
    context.put(IndexedKey.matrixKey("left", 1, 1), 5.0);
    context.put(IndexedKey.matrixKey("left", 1, 2), 6.0);
    MatrixExpression<String> rightMatrix = MatrixExpression.variable("right", 3, 2);
    context.put(IndexedKey.matrixKey("right", 0, 0), 1.0);
    context.put(IndexedKey.matrixKey("right", 0, 1), 2.0);
    context.put(IndexedKey.matrixKey("right", 1, 0), 3.0);
    context.put(IndexedKey.matrixKey("right", 1, 1), 4.0);
    context.put(IndexedKey.matrixKey("right", 2, 0), 5.0);
    context.put(IndexedKey.matrixKey("right", 2, 1), 6.0);
    IMatrixValue<String> product = leftMatrix.times(rightMatrix).evaluate(context);
    assertEquals(2, product.getHeight());
    assertEquals(2, product.getWidth());
    assertEquals(22.0, product.get(0, 0).value(), TOLERANCE);
    assertEquals(28.0, product.get(0, 1).value(), TOLERANCE);
    assertEquals(49.0, product.get(1, 0).value(), TOLERANCE);
    assertEquals(64.0, product.get(1, 1).value(), TOLERANCE);
  }

  @Test
  public void testMatrixAdd() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    MatrixExpression<String> leftMatrix = MatrixExpression.variable("m", 2, 2);
    context.put(IndexedKey.matrixKey("m", 0, 0), 1.0);
    context.put(IndexedKey.matrixKey("m", 0, 1), 2.0);
    context.put(IndexedKey.matrixKey("m", 1, 0), 4.0);
    context.put(IndexedKey.matrixKey("m", 1, 1), 5.0);
    IMatrixValue<String> sum = leftMatrix.plus(leftMatrix).evaluate(context);
    assertEquals(2, sum.getHeight());
    assertEquals(2, sum.getWidth());
    assertEquals(2, sum.get(0, 0).value(), TOLERANCE);
    assertEquals(4, sum.get(0, 1).value(), TOLERANCE);
    assertEquals(8, sum.get(1, 0).value(), TOLERANCE);
    assertEquals(10, sum.get(1, 1).value(), TOLERANCE);
  }

  @Test
  public void testScalarTimesMatrix() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    MatrixExpression<String> matrix = MatrixExpression.variable("m", 2, 2);
    context.put(IndexedKey.matrixKey("m", 0, 0), 1.0);
    context.put(IndexedKey.matrixKey("m", 0, 1), 2.0);
    context.put(IndexedKey.matrixKey("m", 1, 0), 4.0);
    context.put(IndexedKey.matrixKey("m", 1, 1), 5.0);
    ScalarExpression<String> scalar = ScalarExpression.constant(3);
    IMatrixValue<String> product = scalar.times(matrix).evaluate(context);
    assertEquals(2, product.getHeight());
    assertEquals(2, product.getWidth());
    assertEquals(3, product.get(0, 0).value(), TOLERANCE);
    assertEquals(6, product.get(0, 1).value(), TOLERANCE);
    assertEquals(12, product.get(1, 0).value(), TOLERANCE);
    assertEquals(15, product.get(1, 1).value(), TOLERANCE);
  }

  @Test
  public void testTranspose() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    MatrixExpression<String> matrix = MatrixExpression.variable("m", 2, 3);
    context.put(IndexedKey.matrixKey("m", 0, 0), 1.0);
    context.put(IndexedKey.matrixKey("m", 0, 1), 2.0);
    context.put(IndexedKey.matrixKey("m", 0, 2), 3.0);
    context.put(IndexedKey.matrixKey("m", 1, 0), 4.0);
    context.put(IndexedKey.matrixKey("m", 1, 1), 5.0);
    context.put(IndexedKey.matrixKey("m", 1, 2), 6.0);
    IMatrixValue<String> transpose = matrix.transpose().evaluate(context);
    assertEquals(3, transpose.getHeight());
    assertEquals(2, transpose.getWidth());
    assertEquals(1, transpose.get(0, 0).value(), TOLERANCE);
    assertEquals(4, transpose.get(0, 1).value(), TOLERANCE);
    assertEquals(2, transpose.get(1, 0).value(), TOLERANCE);
    assertEquals(5, transpose.get(1, 1).value(), TOLERANCE);
    assertEquals(3, transpose.get(2, 0).value(), TOLERANCE);
    assertEquals(6, transpose.get(2, 1).value(), TOLERANCE);
  }

  @Ignore
  @Test
  public void largeMatrixPerformanceTest() {
    int size = 250;
    MatrixExpression<String> left = MatrixExpression.variable("left", size, size);
    MatrixExpression<String> right = MatrixExpression.variable("right", size, size);
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        context.put(IndexedKey.matrixKey("left", i, j), Math.random());
        context.put(IndexedKey.matrixKey("right", i, j), Math.random());
      }
    }
    // warmup
    left.times(right).evaluate(context);
    long start = System.currentTimeMillis();
    // run
    left.times(right).evaluate(context);
    long end = System.currentTimeMillis();
    System.out.println("Matrix times matrix time millis = " + (end - start));
  }

  @Ignore
  @Test
  public void vectorTimeMatrixPerformanceTest() {
    int size = 1000;
    MatrixExpression<String> left = MatrixExpression.variable("left", 1, size);
    MatrixExpression<String> right = MatrixExpression.variable("right", size, size);
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    for (int i = 0; i < size; ++i) {
      context.put(IndexedKey.matrixKey("left", 0, i), Math.random());
      for (int j = 0; j < size; ++j) {
        context.put(IndexedKey.matrixKey("right", i, j), Math.random());
      }
    }
    // warmup
    for (int i = 0; i < 10; ++i) {
      left.times(right).evaluate(context);
    }
    long start = System.currentTimeMillis();
    // run
    for (int i = 0; i < 10; ++i) {
      left.times(right).evaluate(context);
    }
    long end = System.currentTimeMillis();
    System.out.println("Vector times matrix time millis = " + (end - start));
  }

}
