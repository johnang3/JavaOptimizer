package angland.optimizer.var;

import static org.junit.Assert.assertEquals;

import java.util.HashMap;
import java.util.Map;

import org.junit.Ignore;
import org.junit.Test;

import angland.optimizer.var.matrix.IMatrixValue;
import angland.optimizer.var.scalar.IScalarValue;

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
    Map<IndexedKey<String>, Double> cMap = new HashMap<>();

    cMap.put(IndexedKey.matrixKey("left", 0, 0), 1.0);
    cMap.put(IndexedKey.matrixKey("left", 0, 1), 2.0);
    cMap.put(IndexedKey.matrixKey("left", 0, 2), 3.0);
    cMap.put(IndexedKey.matrixKey("left", 1, 0), 4.0);
    cMap.put(IndexedKey.matrixKey("left", 1, 1), 5.0);
    cMap.put(IndexedKey.matrixKey("left", 1, 2), 6.0);

    cMap.put(IndexedKey.matrixKey("right", 0, 0), 1.0);
    cMap.put(IndexedKey.matrixKey("right", 0, 1), 2.0);
    cMap.put(IndexedKey.matrixKey("right", 1, 0), 3.0);
    cMap.put(IndexedKey.matrixKey("right", 1, 1), 4.0);
    cMap.put(IndexedKey.matrixKey("right", 2, 0), 5.0);
    cMap.put(IndexedKey.matrixKey("right", 2, 1), 6.0);

    Context<String> context = ContextTemplate.simpleContext(cMap);
    IMatrixValue<String> leftMatrix = IMatrixValue.var("left", 2, 3, context);
    IMatrixValue<String> rightMatrix = IMatrixValue.var("right", 3, 2, context);
    IMatrixValue<String> product = leftMatrix.times(rightMatrix);
    assertEquals(2, product.getHeight());
    assertEquals(2, product.getWidth());
    assertEquals(22.0, product.get(0, 0).value(), TOLERANCE);
    assertEquals(28.0, product.get(0, 1).value(), TOLERANCE);
    assertEquals(49.0, product.get(1, 0).value(), TOLERANCE);
    assertEquals(64.0, product.get(1, 1).value(), TOLERANCE);
    IMatrixValue<String> streamedProduct = leftMatrix.streamingTimes(rightMatrix);
    assertEquals(2, streamedProduct.getHeight());
    assertEquals(2, streamedProduct.getWidth());
    assertEquals(22.0, streamedProduct.get(0, 0).value(), TOLERANCE);
    assertEquals(28.0, streamedProduct.get(0, 1).value(), TOLERANCE);
    assertEquals(49.0, streamedProduct.get(1, 0).value(), TOLERANCE);
    assertEquals(64.0, streamedProduct.get(1, 1).value(), TOLERANCE);
  }

  @Test
  public void testMatrixAdd() {
    Map<IndexedKey<String>, Double> cMap = new HashMap<>();
    cMap.put(IndexedKey.matrixKey("m", 0, 0), 1.0);
    cMap.put(IndexedKey.matrixKey("m", 0, 1), 2.0);
    cMap.put(IndexedKey.matrixKey("m", 1, 0), 4.0);
    cMap.put(IndexedKey.matrixKey("m", 1, 1), 5.0);
    Context<String> context = ContextTemplate.simpleContext(cMap);
    IMatrixValue<String> leftMatrix = IMatrixValue.var("m", 2, 2, context);
    IMatrixValue<String> sum = leftMatrix.plus(leftMatrix);
    assertEquals(2, sum.getHeight());
    assertEquals(2, sum.getWidth());
    assertEquals(2, sum.get(0, 0).value(), TOLERANCE);
    assertEquals(4, sum.get(0, 1).value(), TOLERANCE);
    assertEquals(8, sum.get(1, 0).value(), TOLERANCE);
    assertEquals(10, sum.get(1, 1).value(), TOLERANCE);
  }

  @Test
  public void testMatrixVCat() {
    Map<IndexedKey<String>, Double> cMap = new HashMap<>();
    cMap.put(IndexedKey.matrixKey("a", 0, 0), 1.0);
    cMap.put(IndexedKey.matrixKey("a", 0, 1), 2.0);
    cMap.put(IndexedKey.matrixKey("a", 1, 0), 3.0);
    cMap.put(IndexedKey.matrixKey("a", 1, 1), 4.0);
    cMap.put(IndexedKey.matrixKey("b", 0, 0), 5.0);
    cMap.put(IndexedKey.matrixKey("b", 0, 1), 6.0);
    cMap.put(IndexedKey.matrixKey("b", 1, 0), 7.0);
    cMap.put(IndexedKey.matrixKey("b", 1, 1), 8.0);
    Context<String> context = ContextTemplate.simpleContext(cMap);
    IMatrixValue<String> a = IMatrixValue.var("a", 2, 2, context);
    IMatrixValue<String> b = IMatrixValue.var("b", 2, 2, context);
    IMatrixValue<String> ab = a.vCat(b);
    assertEquals(1.0, ab.get(0, 0).value(), TOLERANCE);
    assertEquals(2.0, ab.get(0, 1).value(), TOLERANCE);
    assertEquals(3.0, ab.get(1, 0).value(), TOLERANCE);
    assertEquals(4.0, ab.get(1, 1).value(), TOLERANCE);
    assertEquals(5.0, ab.get(2, 0).value(), TOLERANCE);
    assertEquals(6.0, ab.get(2, 1).value(), TOLERANCE);
    assertEquals(7.0, ab.get(3, 0).value(), TOLERANCE);
    assertEquals(8.0, ab.get(3, 1).value(), TOLERANCE);
  }

  @Test
  public void testChooseAndRandom() {
    Map<IndexedKey<String>, Double> cMap = new HashMap<>();
    int rowCount = 25;
    int forcedIndex = 5;
    int sampleCount = 9;
    for (int i = 0; i < rowCount; ++i) {
      cMap.put(IndexedKey.matrixKey("m", i, 0), (double) i + 1);
    }
    IMatrixValue<String> m =
        IMatrixValue.var("m", rowCount, 1, ContextTemplate.simpleContext(cMap));
    IMatrixValue<String> sample = m.selectAndSampleRows(forcedIndex, sampleCount);
    assertEquals(forcedIndex + 1, sample.get(forcedIndex, 0).value(), TOLERANCE);
    int nonZeroCount = 0;
    for (int i = 0; i < rowCount; ++i) {
      if (i != forcedIndex) {
        double value = sample.get(i, 0).value();
        if (value != 0) {
          ++nonZeroCount;
          assertEquals(i + 1, value, TOLERANCE);
        }
      }
    }
    assertEquals(sampleCount, nonZeroCount);
  }


  @Test
  public void testSampleZero() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    int rowCount = 25;
    int forcedIndex = 5;
    int sampleCount = 0;
    for (int i = 0; i < rowCount; ++i) {
      context.put(IndexedKey.matrixKey("m", i, 0), (double) i + 1);
    }

    IMatrixValue<String> m =
        IMatrixValue.var("m", rowCount, 1, ContextTemplate.simpleContext(context));
    IMatrixValue<String> sample = m.selectAndSampleRows(forcedIndex, sampleCount);
    assertEquals(forcedIndex + 1, sample.get(forcedIndex, 0).value(), TOLERANCE);
    int nonZeroCount = 0;
    for (int i = 0; i < rowCount; ++i) {
      if (i != forcedIndex) {
        double value = sample.get(i, 0).value();
        if (value != 0) {
          ++nonZeroCount;
          assertEquals(i + 1, value, TOLERANCE);
        }
      }
    }
    assertEquals(sampleCount, nonZeroCount);
  }

  @Test
  public void testScalarTimesMatrix() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();

    context.put(IndexedKey.matrixKey("m", 0, 0), 1.0);
    context.put(IndexedKey.matrixKey("m", 0, 1), 2.0);
    context.put(IndexedKey.matrixKey("m", 1, 0), 4.0);
    context.put(IndexedKey.matrixKey("m", 1, 1), 5.0);
    IMatrixValue<String> matrix =
        IMatrixValue.var("m", 2, 2, ContextTemplate.simpleContext(context));
    IScalarValue<String> scalar = IScalarValue.constant(3);
    IMatrixValue<String> product = scalar.times(matrix);
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
    context.put(IndexedKey.matrixKey("m", 0, 0), 1.0);
    context.put(IndexedKey.matrixKey("m", 0, 1), 2.0);
    context.put(IndexedKey.matrixKey("m", 0, 2), 3.0);
    context.put(IndexedKey.matrixKey("m", 1, 0), 4.0);
    context.put(IndexedKey.matrixKey("m", 1, 1), 5.0);
    context.put(IndexedKey.matrixKey("m", 1, 2), 6.0);
    IMatrixValue<String> matrix =
        IMatrixValue.var("m", 2, 3, ContextTemplate.simpleContext(context));
    IMatrixValue<String> transpose = matrix.transpose();
    assertEquals(3, transpose.getHeight());
    assertEquals(2, transpose.getWidth());
    assertEquals(1, transpose.get(0, 0).value(), TOLERANCE);
    assertEquals(4, transpose.get(0, 1).value(), TOLERANCE);
    assertEquals(2, transpose.get(1, 0).value(), TOLERANCE);
    assertEquals(5, transpose.get(1, 1).value(), TOLERANCE);
    assertEquals(3, transpose.get(2, 0).value(), TOLERANCE);
    assertEquals(6, transpose.get(2, 1).value(), TOLERANCE);
  }

  @Test
  public void testColumnProximity() {
    Map<IndexedKey<String>, Double> cMap = new HashMap<>();

    cMap.put(IndexedKey.matrixKey("a", 0, 0), 3.0);
    cMap.put(IndexedKey.matrixKey("a", 0, 1), 6.0);
    cMap.put(IndexedKey.matrixKey("a", 1, 0), 4.0);
    cMap.put(IndexedKey.matrixKey("a", 1, 1), -8.0);
    cMap.put(IndexedKey.matrixKey("b", 0, 0), 0.0);
    cMap.put(IndexedKey.matrixKey("b", 1, 0), 0.0);
    Context<String> c = ContextTemplate.simpleContext(cMap);
    IMatrixValue<String> a = IMatrixValue.var("a", 2, 2, c);
    IMatrixValue<String> b = IMatrixValue.var("b", 2, 1, c);
    IMatrixValue<String> norm = a.columnProximity(b);
    assertEquals(norm.getHeight(), 1);
    assertEquals(norm.getWidth(), 2);
    assertEquals(norm.get(0, 0).value(), 5.0, TOLERANCE);
    assertEquals(norm.get(0, 1).value(), 10.0, TOLERANCE);
  }

  @Test
  public void testMaxIdx() {
    Map<IndexedKey<String>, Double> cMap = new HashMap<>();
    cMap.put(IndexedKey.matrixKey("a", 0, 0), 3.0);
    cMap.put(IndexedKey.matrixKey("a", 1, 0), 6.0);
    cMap.put(IndexedKey.matrixKey("a", 2, 0), 4.0);
    IMatrixValue<String> a = IMatrixValue.var("a", 3, 1, ContextTemplate.simpleContext(cMap));
    IScalarValue<String> maxIdx = a.maxIdx();
    assertEquals(1, maxIdx.value(), TOLERANCE);
  }

  @Ignore
  @Test
  public void largeMatrixPerformanceTest() {
    int size = 250;
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        context.put(IndexedKey.matrixKey("left", i, j), Math.random());
        context.put(IndexedKey.matrixKey("right", i, j), Math.random());
      }
    }
    IMatrixValue<String> left =
        IMatrixValue.var("left", size, size, ContextTemplate.simpleContext(context));
    IMatrixValue<String> right =
        IMatrixValue.var("right", size, size, ContextTemplate.simpleContext(context));
    // warmup
    left.times(right);
    long start = System.currentTimeMillis();
    // run
    left.times(right);
    long end = System.currentTimeMillis();
    System.out.println("Matrix times matrix time millis = " + (end - start));
  }


  @Test
  public void vectorTimesMatrixPerformanceTest() {
    int size = 200;

    Map<IndexedKey<String>, Double> context = new HashMap<>();
    for (int i = 0; i < size; ++i) {
      context.put(IndexedKey.matrixKey("left", 0, i), Math.random());
      for (int j = 0; j < size; ++j) {
        context.put(IndexedKey.matrixKey("right", i, j), Math.random());
      }
    }
    IMatrixValue<String> left =
        IMatrixValue.var("left", 1, size, ContextTemplate.simpleContext(context));
    IMatrixValue<String> right =
        IMatrixValue.var("right", size, size, ContextTemplate.simpleContext(context));
    // warmup
    for (int i = 0; i < 10; ++i) {
      left.times(right);
    }
    long start = System.currentTimeMillis();
    // run
    for (int i = 0; i < 60; ++i) {
      left.times(right);
    }
    long end = System.currentTimeMillis();
    System.out.println("Vector times matrix time millis = " + (end - start));
  }

  @Test
  public void testGetColumn() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.matrixKey("m", 0, 0), 1.0);
    context.put(IndexedKey.matrixKey("m", 0, 1), 2.0);
    context.put(IndexedKey.matrixKey("m", 1, 0), 4.0);
    context.put(IndexedKey.matrixKey("m", 1, 1), 5.0);
    context.put(IndexedKey.matrixKey("m", 2, 0), 7.0);
    context.put(IndexedKey.matrixKey("m", 2, 1), 8.0);
    IMatrixValue<String> matrix =
        IMatrixValue.var("m", 3, 2, ContextTemplate.simpleContext(context));
    IMatrixValue<String> col = matrix.getColumn(IScalarValue.constant(1));
    assertEquals(1, col.getWidth());
    assertEquals(3, col.getHeight());
    assertEquals(2.0, col.get(0, 0).value(), TOLERANCE);
    assertEquals(5.0, col.get(1, 0).value(), TOLERANCE);
  }

  @Test
  public void testSoftmax() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.matrixKey("m", 0, 0), 1.0);
    context.put(IndexedKey.matrixKey("m", 1, 0), 2.0);
    context.put(IndexedKey.matrixKey("m", 2, 0), 4.0);
    context.put(IndexedKey.matrixKey("m", 3, 0), 5.0);
    context.put(IndexedKey.matrixKey("m", 4, 0), 7.0);
    context.put(IndexedKey.matrixKey("m", 5, 0), 8.0);
    IMatrixValue<String> matrix =
        IMatrixValue.var("m", 6, 1, ContextTemplate.simpleContext(context));
    IMatrixValue<String> softmax = matrix.softmax();
    assertEquals(1.0, softmax.elementSumStream().value(), TOLERANCE);
  }


}
