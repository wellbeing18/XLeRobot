# Multi-Task Dataset Strategy - Quick Summary

**Updated:** 2025-11-23
**Document:** COMPLETE_LORA_GUIDE.md v1.1

## Key Decision: Multi-Task from MVP Stage

### Recommended Approach ✅

**MVP Stage (50 episodes):**
```
Pick-and-Place:  30 episodes (60%)
Push/Slide:      20 episodes (40%)
```

**Full Stage (100 episodes):**
```
Pick-and-Place:  40 episodes (40%)
Push/Slide:      30 episodes (30%)
Open Drawer:     15 episodes (15%)
Close Drawer:    15 episodes (15%)
```

## Why Multi-Task?

### Benefits
1. ✅ **Data Efficiency:** Shared skills (reaching, grasping) transfer across tasks
2. ✅ **Better Generalization:** Model learns manipulation primitives, not task memorization
3. ✅ **Early Validation:** Know immediately if multi-task learning works
4. ✅ **Reusable Data:** All 50 MVP episodes contribute to final model

### Tradeoffs
- ⚠️ Takes 1 extra week (5 weeks vs 4 weeks for single-task)
- ⚠️ Slightly more complex setup (2-4 different task environments)
- ⚠️ Per-task success rate 5-10% lower than single-task specialist

## Alternative: Staged Approach (Lower Risk)

If you want to validate pipeline first:

**Phase 1:** 20 episodes pick-and-place (2-3 days)
→ Validate pipeline works

**Phase 2:** +15 pick + 15 push (3-4 days)
→ Validate multi-task learning

**Total:** 50 episodes (35 pick, 15 push)

## What Changed in Guide

### New Sections
1. **Dataset Collection Strategy** (replaces "Dataset Collection")
   - Multi-task learning overview
   - Task distribution recommendations
   - Task-specific recording tips

2. **Multi-Task Analysis** (added to MVP Stage)
   - How to analyze multi-task loss curves
   - Decision scenarios (both tasks learn, one struggles, etc.)

3. **Multi-Task Evaluation** (updated in Evaluation section)
   - Per-task testing protocols (10 trials × 4 tasks)
   - Expected performance by task
   - Zero-shot transfer insights

### Updated Sections
- **Target Dataset Size:** Now shows task distribution
- **Recording Episodes:** Commands for multi-task collection
- **Verifying Dataset:** Shows task metadata structure
- **Expected Performance:** Per-task breakdown instead of overall
- **Timeline:** 5 weeks instead of 4, detailed per-task schedule

## Quick Reference: Data Collection Timeline

**Week 1:**
- Day 1: Setup environments
- Day 2-4: Record 30 pick-and-place episodes
- Day 5-6: Record 20 push/slide episodes
- Day 7: Run MVP training, analyze multi-task learning

**Week 2:**
- Add 10 pick + 10 push episodes

**Week 3:**
- Add 15 open + 15 close drawer episodes

**Week 4:**
- Full training (overnight)
- Multi-task evaluation (40 trials)

**Week 5:**
- Deployment & iteration

## Expected MVP Results

**Scenario A: Multi-task works** ✅
```
Pick loss: 2.0 → 0.6
Push loss: 2.1 → 0.7
```
→ Proceed to full stage

**Scenario B: Primary learns, secondary struggles** ⚠️
```
Pick loss: 2.0 → 0.6
Push loss: 2.1 → 1.9
```
→ Add 10 more push episodes

**Scenario C: Neither learns** ❌
```
Both: > 1.5
```
→ Pipeline issue (not multi-task problem)

## Expected Full Stage Results (100 episodes)

**GR00T Model:**
- Pick-and-place: 70-80% success
- Push/slide: 60-70% success
- Open drawer: 50-60% success
- Close drawer: 50-60% success
- **Overall average: 65-75%**

**Key Insight:** Overall multi-task performance slightly lower than single-task specialist (would be 75-85% for 100 pick-only episodes), but model generalizes better to new tasks.

## Decision Point

**Choose Multi-Task if:**
- ✅ Goal is generalized manipulation capability
- ✅ Want to test multiple tasks in deployment
- ✅ Willing to spend 1 extra week for better generalization

**Choose Single-Task if:**
- ✅ Need fastest path to deployment (4 weeks)
- ✅ Only care about one specific task
- ✅ Want highest possible success rate on that task

## Recommendation

**Go with multi-task MVP (30 pick + 20 push)** because:
1. Your stated goal is data efficiency across tasks (from DATA_COLLECTION_RESEARCH_REPORT)
2. Early validation saves time if multi-task doesn't work
3. 50 episodes still qualifies as "minimal" dataset
4. All data reusable for full stage
5. Push task is easy to add (shares reaching skills)

The 1-week time investment is worth it for validating your core hypothesis: that multi-task learning improves data efficiency and generalization.
