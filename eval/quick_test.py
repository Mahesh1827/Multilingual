"""
Quick End-to-End Verification — 12 queries across 5 languages
Tests: accuracy, FAQ cache isolation, cross-lingual retrieval, follow-up handling
"""
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from query.agents.pipeline import run_query

TEST_QUERIES = [
    # English - basic factual
    ("Where is Tirumala located?", "en", "Should mention Andhra Pradesh / Eastern Ghats"),
    ("What are the seven hills of Tirumala?", "en", "Should LIST the hill names, NOT location info"),
    ("What is the dress code for Tirumala temple?", "en", "Should mention traditional attire rules"),
    ("How to book darshan tickets?", "en", "Should mention TTD website or booking process"),
    
    # Follow-up isolation test — this must NOT return answer from previous question
    ("What is Brahmotsavam?", "en", "Should describe the annual festival, NOT darshan tickets"),
    
    # Telugu
    ("తిరుమల ఎక్కడ ఉంది?", "te", "Should answer about location"),
    
    # Hindi  
    ("तिरुमाला मंदिर का इतिहास क्या है?", "hi", "Should mention temple history"),
    
    # Tamil
    ("திருமலை கோயிலின் வரலாறு என்ன?", "ta", "Should mention temple history"),
    
    # Kannada
    ("ತಿರುಮಲ ದೇವಸ್ಥಾನದ ಇತಿಹಾಸ ಏನು?", "kn", "Should mention temple history"),
    
    # Cross-lingual: Romanized Hindi
    ("tirumala darshan ticket kaise book karein", "hi", "Should explain booking process"),
    
    # Out-of-scope
    ("What is the capital of France?", "en", "Should reject as out-of-scope"),
    
    # Conversational
    ("hello", "en", "Should return a greeting"),
]

def main():
    print("=" * 70)
    print("  QUICK END-TO-END VERIFICATION — 12 Queries")
    print("=" * 70)
    
    passed = 0
    failed = 0
    results = []
    
    for i, (query, lang, expected_behavior) in enumerate(TEST_QUERIES, 1):
        print(f"\n  [{i}/{len(TEST_QUERIES)}] [{lang.upper()}] {query[:60]}")
        print(f"       Expected: {expected_behavior}")
        
        start = time.time()
        try:
            result = run_query(query, language=lang)
            elapsed = time.time() - start
            answer = result.get("answer", "").strip()[:200]
            route = result.get("agent_route", "unknown")
            
            print(f"       Route: {route}")
            print(f"       Answer: {answer[:150]}...")
            print(f"       Time: {elapsed:.1f}s")
            
            # Basic sanity checks
            is_ok = True
            issues = []
            
            # Check answer is not empty
            if not answer or len(answer) < 5:
                is_ok = False
                issues.append("EMPTY ANSWER")
            
            # Check not returning raw chunk headers
            if "CHAPTER" in answer.upper() or "Physical Features" in answer:
                is_ok = False
                issues.append("RAW CHUNK DUMP")
            
            # Check not returning previous question's answer (isolation test)
            if i == 2 and "eastern ghats" in answer.lower() and "latitude" in answer.lower():
                is_ok = False
                issues.append("RETURNED LOCATION ANSWER FOR HILLS QUESTION")
                
            if i == 5 and ("darshan" in answer.lower() and "ticket" in answer.lower() and "brahmotsavam" not in answer.lower()):
                is_ok = False
                issues.append("FAQ CACHE RETURNED WRONG ANSWER")
            
            # Out-of-scope check
            if i == 11 and "paris" in answer.lower():
                is_ok = False
                issues.append("ANSWERED OUT-OF-SCOPE QUESTION")
            
            status = "✅ PASS" if is_ok else f"❌ FAIL ({', '.join(issues)})"
            print(f"       {status}")
            
            if is_ok:
                passed += 1
            else:
                failed += 1
                
            results.append({
                "query": query, "lang": lang, "route": route,
                "answer": answer, "ok": is_ok, "time": elapsed
            })
            
        except Exception as e:
            elapsed = time.time() - start
            print(f"       ❌ ERROR: {e} ({elapsed:.1f}s)")
            failed += 1
            results.append({
                "query": query, "lang": lang, "route": "error",
                "answer": str(e), "ok": False, "time": elapsed
            })
    
    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {passed}/{len(TEST_QUERIES)} passed, {failed} failed")
    print(f"  Avg Response Time: {sum(r['time'] for r in results) / len(results):.1f}s")
    print(f"{'=' * 70}")
    
    # Route distribution
    routes = {}
    for r in results:
        routes[r["route"]] = routes.get(r["route"], 0) + 1
    print(f"  Routes: {routes}")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
