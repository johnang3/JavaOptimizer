package angland.optimizer.nn;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

public class FeedForwardLayerTest {

  @Test
  public void testStreamKeys() {
    assertEquals(30, (int) FeedForwardLayer.getVarKeys("w", "b", 5, 5).count());
  }
}
