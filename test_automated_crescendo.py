#!/usr/bin/env python3
"""
Test script for the automated crescendo attack pipeline.

This script validates the pipeline functionality with a limited test run
before executing full-scale automated attacks.
"""

import sys
import json
from automated_crescendo import (
    AutomatedCrescendoPipeline,
    HarmfulCategory,
    EscalationStrategy,
    AttackTemplateLibrary
)


def test_template_library():
    """Test the attack template library functionality."""
    print("Testing Attack Template Library...")
    
    library = AttackTemplateLibrary()
    
    # Test getting templates
    violence_template = library.get_template(HarmfulCategory.VIOLENCE, EscalationStrategy.GRADUAL)
    assert violence_template is not None, "Failed to get violence template"
    
    # Test getting templates by category
    violence_templates = library.get_templates_by_category(HarmfulCategory.VIOLENCE)
    assert len(violence_templates) > 0, "No violence templates found"
    
    # Test getting all categories
    categories = library.get_all_categories()
    assert len(categories) > 0, "No categories found"
    
    print("✅ Template library tests passed")


def test_safety_evaluator():
    """Test the safety evaluation functionality."""
    print("Testing Safety Evaluator...")
    
    from automated_crescendo import SafetyEvaluator
    evaluator = SafetyEvaluator()
    
    # Test refusal detection
    refusal_response = "I can't help with that request as it could be harmful."
    cooperative_response = "Here are the detailed steps you requested..."
    
    library = AttackTemplateLibrary()
    violence_template = library.get_template(HarmfulCategory.VIOLENCE, EscalationStrategy.GRADUAL)
    
    refusal_analysis = evaluator.analyze_response(refusal_response, violence_template)
    cooperative_analysis = evaluator.analyze_response(cooperative_response, violence_template)
    
    assert refusal_analysis['refusal_level'] > 0.5, "Failed to detect refusal"
    assert cooperative_analysis['refusal_level'] < 0.5, "False refusal detection"
    
    print("✅ Safety evaluator tests passed")


def test_single_attack():
    """Test executing a single automated attack."""
    print("Testing Single Attack Execution...")
    
    try:
        # Initialize pipeline
        pipeline = AutomatedCrescendoPipeline()
        
        # Get a template for testing
        library = AttackTemplateLibrary()
        template = library.get_template(HarmfulCategory.VIOLENCE, EscalationStrategy.GRADUAL)
        
        if template is None:
            print("❌ No template found for testing")
            return False
        
        # Execute attack with limited turns for testing
        template.max_turns = 2  # Limit for quick testing
        result = pipeline.execute_attack(template, "test_attack_001")
        
        # Validate result
        assert result is not None, "Attack execution returned None"
        assert len(result.conversation_turns) > 0, "No conversation turns recorded"
        assert result.template_id == "test_attack_001", "Template ID mismatch"
        
        print(f"✅ Single attack test passed - {len(result.conversation_turns)} turns executed")
        return True
        
    except Exception as e:
        print(f"❌ Single attack test failed: {str(e)}")
        return False


def test_batch_processing():
    """Test limited batch processing functionality."""
    print("Testing Limited Batch Processing...")
    
    try:
        # Initialize pipeline
        pipeline = AutomatedCrescendoPipeline()
        
        # Run limited batch test (only violence category, 1 attempt)
        results = pipeline.run_batch_testing(
            categories=[HarmfulCategory.VIOLENCE],
            strategies=[EscalationStrategy.GRADUAL],
            num_attempts=1
        )
        
        assert len(results) > 0, "No results from batch testing"
        
        # Test saving results
        output_file, summary_file = pipeline.save_results(results, "test_results.json")
        
        # Verify files were created
        import os
        assert os.path.exists(output_file), "Output file not created"
        assert os.path.exists(summary_file), "Summary file not created"
        
        print(f"✅ Batch processing test passed - {len(results)} results generated")
        print(f"   Files created: {output_file}, {summary_file}")
        
        # Clean up test files
        os.remove(output_file)
        os.remove(summary_file)
        
        return True
        
    except Exception as e:
        print(f"❌ Batch processing test failed: {str(e)}")
        return False


def run_quick_demo():
    """Run a quick demonstration of the pipeline."""
    print("\n" + "="*60)
    print("QUICK DEMONSTRATION")
    print("="*60)
    
    try:
        pipeline = AutomatedCrescendoPipeline()
        
        # Show available categories and strategies
        library = AttackTemplateLibrary()
        categories = library.get_all_categories()
        
        print(f"Available categories: {[c.value for c in categories]}")
        print(f"Available strategies: {[s.value for s in EscalationStrategy]}")
        
        # Run one attack from each available category
        demo_results = []
        for category in categories[:2]:  # Limit to first 2 categories for demo
            template = library.get_template(category, EscalationStrategy.GRADUAL)
            if template:
                print(f"\nExecuting demo attack: {category.value}")
                template.max_turns = 3  # Limit turns for demo
                result = pipeline.execute_attack(template, f"demo_{category.value}")
                demo_results.append(result)
        
        # Print summary
        if demo_results:
            pipeline.print_summary(demo_results)
        
        print("\n✅ Demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        return False


def main():
    """Run all tests for the automated crescendo pipeline."""
    print("Automated Crescendo Pipeline - Test Suite")
    print("="*50)
    
    tests = [
        ("Template Library", test_template_library),
        ("Safety Evaluator", test_safety_evaluator),
        ("Single Attack", test_single_attack),
        ("Batch Processing", test_batch_processing),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {str(e)}")
    
    print(f"\n{'='*50}")
    print(f"TEST SUMMARY: {passed}/{total} tests passed")
    print(f"{'='*50}")
    
    if passed == total:
        print("🎉 All tests passed! Pipeline is ready for use.")
        
        # Ask if user wants to see demo
        try:
            demo_choice = input("\nWould you like to see a quick demo? (y/n): ").strip().lower()
            if demo_choice in ['y', 'yes']:
                run_quick_demo()
        except (KeyboardInterrupt, EOFError):
            print("\nSkipping demo.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
