package angland.optimizer.nn;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

public class FeedForwardLayerTest {

  @Test
  public void testStreamKeys() {
    FeedForwardLayer<String> layer =
        new FeedForwardLayer<String>(5, 5, sv -> sv.sigmoid(), "w", "b");
    assertEquals(30, (int) layer.getVarKeys().count());
  }
}
