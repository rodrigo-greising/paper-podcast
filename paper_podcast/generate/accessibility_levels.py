"""
Accessibility level system for generating podcast episodes tailored to different expertise levels.

This module defines the 5 accessibility levels and their corresponding system prompts
for generating podcast content appropriate for different audience backgrounds.
"""

from typing import Dict, NamedTuple


class AccessibilityLevel(NamedTuple):
    """Configuration for a specific accessibility level."""
    level: int
    name: str
    description: str
    target_audience: str
    system_prompt: str
    file_suffix: str


# Define the 5 accessibility levels
ACCESSIBILITY_LEVELS: Dict[int, AccessibilityLevel] = {
    1: AccessibilityLevel(
        level=1,
        name="Expert",
        description="Deep technical analysis for domain experts",
        target_audience="AI/ML researchers, PhD students, domain experts",
        file_suffix="expert",
        system_prompt=(
            "You are generating a comprehensive monthly review podcast episode focusing on a specific AI research topic. "
            "This is a deep technical dive for an EXPERT AUDIENCE with extensive domain knowledge. "
            "CRITICAL REQUIREMENTS: "
            "1) MONTHLY CONTEXT: Frame this as 'This Month in [Topic]' - discuss trends, breakthroughs, and developments from the entire month. "
            "2) MAXIMUM TECHNICAL DEPTH: Cover methodologies, architectures, experimental results, algorithmic innovations, mathematical formulations, and implementation details. "
            "3) COMPREHENSIVE COVERAGE: Discuss multiple papers that represent different approaches or sub-areas within the topic. "
            "4) TREND ANALYSIS: Identify patterns, emerging directions, theoretical implications, and how this month's work builds on or diverges from previous research. "
            "5) CRITICAL EVALUATION: Analyze strengths, limitations, computational complexity, theoretical guarantees, and potential impact. "
            "6) CITATIONS: Include precise inline citations [arXiv:ID] after each technical claim. "
            "7) STRUCTURE: Organize by themes or approaches, not just individual papers. Show theoretical connections between papers. "
            "8) LENGTH: Generate substantial content - aim for 15-25 minutes of detailed technical discussion. "
            "9) EXPERT PERSPECTIVE: Use advanced terminology, discuss mathematical proofs, algorithmic complexity, and assume deep background knowledge. "
            "The hosts should demonstrate expertise at the cutting edge and provide insights that only experienced researchers would appreciate."
        )
    ),

    2: AccessibilityLevel(
        level=2,
        name="ML Expert",
        description="Technical discussion for machine learning experts",
        target_audience="ML engineers, data scientists, CS graduate students",
        file_suffix="ml_expert",
        system_prompt=(
            "You are generating a comprehensive monthly review podcast episode focusing on a specific AI research topic. "
            "This is a technical discussion for MACHINE LEARNING EXPERTS with strong ML background but may not be domain specialists. "
            "CRITICAL REQUIREMENTS: "
            "1) MONTHLY CONTEXT: Frame this as 'This Month in [Topic]' - discuss trends, breakthroughs, and developments from the entire month. "
            "2) TECHNICAL DEPTH: Cover methodologies, architectures, experimental results, and key algorithmic innovations with clear explanations. "
            "3) COMPREHENSIVE COVERAGE: Discuss multiple papers that represent different approaches or sub-areas within the topic. "
            "4) TREND ANALYSIS: Identify patterns, emerging directions, and how this month's work builds on established ML principles. "
            "5) CRITICAL EVALUATION: Analyze strengths, limitations, practical implications, and potential applications. "
            "6) CITATIONS: Include precise inline citations [arXiv:ID] after each technical claim. "
            "7) STRUCTURE: Organize by themes or approaches, showing how papers relate to established ML concepts. "
            "8) LENGTH: Generate substantial content - aim for 15-25 minutes of technical discussion. "
            "9) ML EXPERT PERSPECTIVE: Use standard ML terminology, explain novel concepts in relation to known techniques, discuss practical implementation considerations. "
            "Briefly explain domain-specific terms but assume strong foundation in machine learning, statistics, and optimization."
        )
    ),

    3: AccessibilityLevel(
        level=3,
        name="Software Developer",
        description="Technical overview for experienced programmers",
        target_audience="Software engineers, CS undergraduates, technical professionals",
        file_suffix="developer",
        system_prompt=(
            "You are generating a monthly review podcast episode focusing on a specific AI research topic. "
            "This is a technical overview for SOFTWARE DEVELOPERS with programming experience but limited ML background. "
            "CRITICAL REQUIREMENTS: "
            "1) MONTHLY CONTEXT: Frame this as 'This Month in [Topic]' - discuss trends, breakthroughs, and developments from the entire month. "
            "2) ACCESSIBLE TECHNICAL DEPTH: Cover key methodologies and results, explaining ML concepts in programming terms when possible. "
            "3) COMPREHENSIVE COVERAGE: Discuss multiple papers, focusing on practical innovations and algorithmic approaches. "
            "4) TREND ANALYSIS: Identify patterns and emerging directions, relating them to software development practices. "
            "5) PRACTICAL EVALUATION: Analyze practical implications, potential applications, and implementation challenges. "
            "6) CITATIONS: Include inline citations [arXiv:ID] after major claims. "
            "7) STRUCTURE: Organize by themes, showing relationships between different approaches and existing programming concepts. "
            "8) LENGTH: Generate substantial content - aim for 12-20 minutes of accessible technical discussion. "
            "9) DEVELOPER PERSPECTIVE: Explain ML concepts using programming analogies, discuss algorithmic complexity in familiar terms, mention relevant frameworks/libraries. "
            "Define machine learning terminology clearly and relate new concepts to familiar programming patterns."
        )
    ),

    4: AccessibilityLevel(
        level=4,
        name="Technical Professional",
        description="High-level technical discussion for STEM professionals",
        target_audience="Engineers, scientists, technical managers, STEM graduates",
        file_suffix="technical",
        system_prompt=(
            "You are generating a monthly review podcast episode focusing on a specific AI research topic. "
            "This is a high-level technical discussion for TECHNICAL PROFESSIONALS with STEM background but minimal programming/ML experience. "
            "CRITICAL REQUIREMENTS: "
            "1) MONTHLY CONTEXT: Frame this as 'This Month in [Topic]' - discuss trends, breakthroughs, and developments from the entire month. "
            "2) CONCEPTUAL DEPTH: Focus on key innovations, research directions, and practical implications rather than implementation details. "
            "3) BROAD COVERAGE: Discuss multiple papers, emphasizing breakthrough results and real-world applications. "
            "4) TREND ANALYSIS: Identify major patterns and directions, explaining significance in broader scientific context. "
            "5) PRACTICAL EVALUATION: Analyze potential applications, limitations, and impact on various industries. "
            "6) CITATIONS: Include inline citations [arXiv:ID] for major claims. "
            "7) STRUCTURE: Organize by impact themes, showing how research connects to practical applications. "
            "8) LENGTH: Generate substantial content - aim for 10-18 minutes of accessible discussion. "
            "9) PROFESSIONAL PERSPECTIVE: Use clear technical language, explain AI/ML concepts using scientific analogies, focus on results and implications. "
            "Define all AI/ML terminology clearly and emphasize practical significance over technical implementation."
        )
    ),

    5: AccessibilityLevel(
        level=5,
        name="Layman",
        description="General audience explanation with minimal technical jargon",
        target_audience="General public, business professionals, curious non-technical listeners",
        file_suffix="layman",
        system_prompt=(
            "You are generating a monthly review podcast episode focusing on a specific AI research topic. "
            "This is an accessible discussion for a GENERAL AUDIENCE with no technical background. "
            "CRITICAL REQUIREMENTS: "
            "1) MONTHLY CONTEXT: Frame this as 'This Month in [Topic]' - discuss what happened in AI research this month and why it matters. "
            "2) ACCESSIBLE EXPLANATIONS: Focus on breakthrough results, real-world applications, and societal implications using everyday language. "
            "3) STORYTELLING APPROACH: Discuss key papers through narrative, emphasizing human impact and practical benefits. "
            "4) TREND ANALYSIS: Identify major developments and explain their significance for everyday life and society. "
            "5) PRACTICAL IMPLICATIONS: Focus on how these advances might affect people's lives, businesses, and society. "
            "6) MINIMAL CITATIONS: Reference papers naturally in conversation without formal citation format. "
            "7) STRUCTURE: Organize by real-world impact themes, showing how research connects to familiar concepts. "
            "8) LENGTH: Generate engaging content - aim for 8-15 minutes of accessible discussion. "
            "9) GENERAL AUDIENCE PERSPECTIVE: Avoid jargon entirely, use analogies and metaphors, explain everything in plain English. "
            "Make AI research relatable and exciting for people who may have heard terms like 'machine learning' but don't understand the technical details."
        )
    )
}


def get_accessibility_level(level: int) -> AccessibilityLevel:
    """Get accessibility level configuration by level number."""
    if level not in ACCESSIBILITY_LEVELS:
        raise ValueError(f"Invalid accessibility level: {level}. Must be 1-5.")
    return ACCESSIBILITY_LEVELS[level]


def get_all_levels() -> Dict[int, AccessibilityLevel]:
    """Get all accessibility level configurations."""
    return ACCESSIBILITY_LEVELS.copy()


def validate_level(level: int) -> bool:
    """Validate if the given level is supported."""
    return level in ACCESSIBILITY_LEVELS