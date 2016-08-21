package angland.optimizer.textpredictor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class TokenBiMap {

  private final List<String> idxToToken;
  private final Map<String, Integer> tokenToIdx = new HashMap<>();
  private final int unkIdx;

  public TokenBiMap(List<String> idxToToken, String unkSymbol) {
    this.idxToToken = idxToToken;
    for (int i = 0; i < idxToToken.size(); ++i) {
      tokenToIdx.put(idxToToken.get(i), i);
    }
    this.unkIdx = tokenToIdx.get(unkSymbol);
  }

  public String getToken(int idx) {
    return idxToToken.get(idx);
  }

  public int getIdx(String token) {
    return tokenToIdx.getOrDefault(token.toLowerCase(), unkIdx);
  }

  public int getUnkIdx() {
    return unkIdx;
  }


  public static List<String> tokenize(String line) {

    Matcher matcher = Pattern.compile("\\w+|[.,!?:\\'\\\"]").matcher(line);
    ArrayList<String> tokens = new ArrayList<>();
    while (matcher.find()) {
      tokens.add(matcher.group().toLowerCase());
    }
    return tokens;
  }



}
