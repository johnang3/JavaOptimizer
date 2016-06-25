package angland.optimizer.matrix;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import angland.optimizer.var.MatrixOld;


public class SparseMatrixTest {

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
    MatrixOld<Integer, Integer> left = new MatrixOld<>();
    left.put(0, 0, 1);
    left.put(0, 1, 2);
    left.put(0, 2, 3);
    left.put(1, 0, 4);
    left.put(1, 1, 5);
    left.put(1, 2, 6);
    MatrixOld<Integer, Integer> right = new MatrixOld<>();
    right.put(0, 0, 1);
    right.put(0, 1, 2);
    right.put(1, 0, 3);
    right.put(1, 1, 4);
    right.put(2, 0, 5);
    right.put(2, 1, 6);
    MatrixOld<Integer, Integer> product = MatrixOld.multiply(left, right);
    assertEquals(4, product.getAllEntries().count());
    assertEquals(22.0, product.get(0, 0), TOLERANCE);
    assertEquals(28.0, product.get(0, 1), TOLERANCE);
    assertEquals(49.0, product.get(1, 0), TOLERANCE);
    assertEquals(64.0, product.get(1, 1), TOLERANCE);
  }

}
