package angland.optimizer.saver;

import static org.junit.Assert.assertEquals;

import java.util.HashMap;
import java.util.Map;

import org.junit.Ignore;
import org.junit.Test;

import angland.optimizer.var.IndexedKey;

public class StringContextTest {

  @Ignore
  @Test
  public void test() {
    Map<IndexedKey<String>, Double> ctx = new HashMap<>();
    for (int i = 0; i < 5; ++i) {
      for (int j = 0; j < 5; ++j) {
        ctx.put(new IndexedKey<>("a", i, j), -(double) i * i);
      }
    }
    StringContext.saveContext(ctx, "test");
    Map<IndexedKey<String>, Double> loaded = StringContext.loadContext("test");
    assertEquals(ctx, loaded);
  }

}
