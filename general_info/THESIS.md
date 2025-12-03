SkillHighlight: A BERT-Based Context-Aware Keyword Highlighter for CS/IT Résumé Screening
by
Adriano, Evance Gabrielle C.
Alejandro, Art Genzen A.
Feliciano, Samuel M.












INTRODUCTION
1.1 Project Context
In today’s competitive labor market, the résumé serves as the first gateway between job seekers and employers. Yet, students and fresh graduates often struggle to communicate their competencies effectively, limiting their chances of securing employment. Zakaria et al. (2023) observed that many students face difficulties in aligning their qualifications with employer expectations, noting that guided résumé preparation can significantly improve how these skills are presented. This finding illustrates that even capable applicants may fail to capture the attention of recruiters if their résumés do not meet professional standards.
On the employer side, the challenge is equally pressing. With hundreds of applications submitted for a single role, recruiters increasingly depend on applicant tracking systems (ATS) to manage the volume. However, these systems are not without flaws. Johnson (2023) reported that nearly 88% of employers acknowledge the risk of overlooking highly qualified candidates simply because their résumés are not “ATS-friendly.” Such limitations result in organizations unintentionally screening out strong applicants whose skills and experiences are not formatted in ways that the system can process.
Taken together, the issues reveal a twofold problem: students and fresh graduates face difficulties in showcasing their qualifications effectively, while employers risk missing valuable talent due to the technological constraints of ATS.
Recent independent evidence reinforces the danger: a 2021 Harvard Business School / Accenture study of 2275 employers and 8720 “hidden workers” found that 88% of hiring executives believe their ATS filters out high-skill applicants who could perform the job, and the figure rises to 94% for middle-skill roles (Fuller et al., 2021). The same report concludes that the filters are so aggressive that 27 million U.S. workers are effectively “hidden” from recruiters, not because they lack ability, but because their résumés do not satisfy the exact keyword or formatting rules hard-coded into the ATS.
Such a gap calls for research that tackles résumé writing difficulties faced by students and fresh graduates, while also accounting for the changing demands of recruitment systems. The researchers are motivated by the intent to help young applicants showcase their abilities more effectively and to lessen the obstacles imposed by automated hiring filters.


1.2 Purpose and Description
The purpose of this study is to develop an effective system, SkillHighlight, for identifying and presenting the skill sets of applicants—particularly students and fresh graduates—in a way that aligns with employer expectations. The system emphasizes both hard skills (such as technical abilities, certifications, and specialized knowledge) and soft skills (such as communication, teamwork, and adaptability), while also highlighting relevant keywords frequently sought by recruiters. By doing so, the study supports applicants in presenting their qualifications clearly and authentically, helping them demonstrate genuine competencies rather than relying on buzzwords or redundant terms.
At the same time, the study addresses challenges faced by employers and HR professionals in reviewing large volumes of résumés. By automating the identification of key skills and relevant information, SkillHighlight enhances efficiency and accuracy in the screening process, allowing employers to quickly focus on qualified candidates. This dual approach promotes fairness, clarity, and efficiency in recruitment, benefiting both job seekers and employers. Ultimately, the project aims to empower applicants to showcase their strengths confidently while enabling recruiters to make informed, unbiased hiring decisions.


1.3 Statement of the Problem
In today’s labor market, a résumé is still the ticket to a first interview, but its success depends more on whether a computer can read it than on what the applicant has actually done. Commercial applicant-tracking systems (ATS) are built to save time: they look for exact keywords, steady work dates, and standard headings. If a résumé uses a different font, has a six-month gap, or misses one key word, the system deletes it before any person sees it. Most students and new graduates do not know these rules, so their real skills can be ignored just because the format is wrong. At the same time, employers lose access to good candidates and may leave jobs unfilled.
This study will try to solve that mismatch by answering four questions:
What small formatting or word choices cause ATS programs to skip otherwise qualified student résumés?
How do job ads and ATS settings shape the way students write and design their résumés?
How does heavy ATS automation hide skilled applicants, and how can that loss be measured?
What simple, practical tips can help students write résumés that still sound like them but also pass the computer scan?
1.4 Research Objectives
1.4.1 General Objective
To develop SkillHighlight: Optimizing Résumé Screening with BERT-Assisted Keyword Highlighting, a system designed to automate résumé keyword analysis, provide self-promotion scoring, and improve alignment between applicants’ skills and employer expectations using BERT-driven natural language processing and machine learning models.
1.4.2 Specific Objectives
To achieve the general objective, this study aims to:
Analyze existing résumé screening challenges by reviewing related literature, identifying limitations in Applicant Tracking Systems (ATS), and the shortcomings of manual keyword-based skill evaluation.


Design the system architecture to incorporate BERT-based contextual keyword highlighting, ATS-aware scoring indicators, and automated skill category detection (hard skills, soft skills, and recruiter-preferred traits).


Determine the appropriate programming tools and algorithms required for the implementation of BERT embeddings and K-Nearest Neighbors (KNN) classification for sentence-level self-promotion scoring and contextual keyword detection.


Develop the system by coding functionalities that automatically highlight relevant keywords, compute keyword composition, analyze sentence-level self-promotion, and simplify résumé screening for job seekers and employers.


Integrate meaningful keyword verification using self-promotion scoring to ensure that highlighted skills represent authentic competencies rather than redundant, irrelevant, or buzzword-dependent content.


Test and evaluate the system using sample résumés to assess the accuracy of contextual keyword highlighting, keyword composition ratios, self-promotion scoring, and the overall effectiveness and efficiency of the résumé screening process.


1.5 Significance of the Study
This study addresses a persistent challenge among job seekers and recruiters in an increasingly competitive employment landscape. For students and fresh graduates, effectively conveying their skills and experiences in a résumé can be difficult, often resulting in missed opportunities. Through SkillHighlight, applicants gain an accessible tool that identifies in-demand skills, highlights strengths, and promotes genuine self-presentation instead of overreliance on generic buzzwords. By providing feedback on skill balance and self-promotion, the system assists applicants in improving résumé quality prior to submission.
For recruiters and human resource professionals, SkillHighlight streamlines résumé evaluation by automatically detecting and highlighting relevant skills. This reduces manual screening time and helps ensure that competent candidates are not overlooked due to keyword mismatches or inconsistent résumé formatting. Unlike systems such as Navarro’s (2025) JustScreen, which relied primarily on spaCy and RAKE keyword extraction, this project employs a more advanced architecture by combining BERT contextual embeddings with KNN classification and heuristic scoring to deliver deeper semantic understanding and fairer evaluation of résumé content.
Furthermore, this study contributes to the research community by providing a foundation for the development of fair, scalable, and context-aware résumé analysis tools. Guided by insights from Mehrabi et al. (2022) on fairness and bias mitigation in machine learning, this research demonstrates the importance of designing résumé-screening systems that enhance efficiency while promoting fairness and reliability in recruitment practices. Ultimately, SkillHighlight benefits job seekers, employers, and researchers by bridging longstanding gaps in résumé analysis and enabling more accurate and equitable hiring outcomes.
1.6 Scope and Limitations
The scope of this study is limited to the development of a context-aware keyword highlighting and self-promotion analysis system for résumés in PDF, DOCX, and TXT formats. The system integrates BERT-based contextual embeddings with a K-Nearest Neighbors (KNN) classification model to analyze résumé content, identify and highlight relevant skills, and generate sentence-level self-promotion scoring. The study focuses solely on keyword extraction, context-aware highlighting, and self-promotion evaluation, rather than full applicant tracking system (ATS) functions such as résumé ranking, profile matching, or automated hiring decisions.
As for the limitations, the system does not include OCR capabilities, and therefore cannot process scanned or image-based résumés. The keyword-matching and contextual analysis pipeline currently supports only English-language résumés, limiting usability in multilingual applications. Additionally, the model was trained and evaluated primarily on résumés from the computer science and information technology domains; therefore, performance and keyword coverage may vary across other professional fields. Finally, while the system supports résumé review and skill identification, hiring decisions remain the responsibility of employers and recruitment personnel, and the tool is not intended to replace human judgment in candidate selection.


REVIEW OF RELATED LITERATURE
In today’s job market, résumés remain one of the most important tools for both applicants and employers. For students and fresh graduates, preparing an effective résumé is often a challenge, as they may struggle to communicate their skills in ways that match employer expectations. At the same time, recruiters face the difficulty of reviewing large volumes of applications, often relying on Applicant Tracking Systems (ATS) to streamline the process. While these systems improve efficiency, they also risk overlooking qualified candidates if résumés are not formatted or written in an ATS-friendly manner.
This research introduces SkillHighlight, a system designed to improve résumé screening by using advanced Natural Language Processing (NLP) techniques, particularly BERT-based keyword highlighting. The goal is twofold: to help applicants, especially students and fresh graduates, showcase their skills more effectively, and to support recruiters in identifying qualified candidates more efficiently and fairly. By bridging the gap between résumé preparation and automated screening technologies, SkillHighlight seeks to make the recruitment process clearer, more reliable, and more equitable.

2.1 Resume Parsing and Keyword Extraction
Resume parsing and keyword extraction are essential for transforming unstructured resumes into analyzable text. Parsing converts documents like PDFs or Word files into plain text, while keyword extraction identifies relevant skills, qualifications, and experiences for matching candidates to job requirements.
Several studies highlight different approaches. Sanyal et al. (2017, as cited in Patlolla et al., 2023) and Daryania et al. (2020, as cited in Patlolla et al., 2023) developed NLP-based systems that extract keywords and generate relevance scores by comparing resumes with job descriptions. Vaidya et al. (2015, as cited in Patlolla et al., 2023) emphasized preprocessing steps, such as stop-word removal and Soundex algorithms, to improve keyword matching, while Nawander et al. (2022, as cited in Patlolla et al., 2023) combined NLP and Streamlit modules to parse and rank PDF resumes.
Other researchers extended parsing techniques with ML. Pokharel (2018, as cited in Patlolla et al., 2023) and Bhor et al. (2021, as cited in Patlolla et al., 2023) focused on extracting structured sections and key details, while Pravesh et al. (2022, as cited in Patlolla et al., 2023) and Gupta et al. (2022, as cited in Patlolla et al., 2023) applied ML algorithms to improve ranking and candidate matching. Thakur and Goyal (n.d., as cited in Patlolla et al., 2023) introduced automated resume classification to reduce manual screening effort.
These studies show that parsing and keyword extraction, combined with NLP and ML, effectively identify important candidate information, rank resumes, and highlight skills—forming the foundation for tools like the SkillHighlight.

2.2 Resume Parsing and Keyword Extraction
Simply reading resumes is rarely enough to find the right candidate. Employers are not just looking at work history—they want to quickly see whether a person has the specific skills and experiences that match the job. This is where keyword highlighting and scoring become essential, helping recruiters focus on what truly matters.
Research shows that analyzing resumes through keyword scoring can make a big difference. Daryania et al. (2020, as cited in Patlolla et al., 2023) explain that comparing keywords in resumes with job descriptions and assigning relevance scores helps rank candidates more effectively. Sanyal et al. (2017, as cited in Patlolla et al., 2023) similarly highlight how NLP can identify key skills and measure how well a candidate fits a role. Even preprocessing steps like stop-word removal and counting keyword frequency, as noted by Vaidya et al. (2015, as cited in Patlolla et al., 2023), improve the accuracy of matching skills to job requirements.
Building on these ideas, SkillHighlight automatically scans résumés for relevant skill keywords while going beyond simple keyword counting. Instead of scoring résumés based on frequency alone, the system evaluates the context of each sentence using BERT embeddings and deep-learning classifiers to measure how effectively applicants communicate their skills and achievements. This provides recruiters with a clearer, more meaningful snapshot of candidate strengths, helping streamline the selection process and support fairer evaluation.


2.3 Self-Promotion in Resumes

	Research shows that applicants commonly use self-promotional strategies in resumes to highlight competence, motivation, and job-related skills and qualifications (Godfrey, Jones, & Lord, 1986; Higgins & Judge, 2004). Waung et al. (2015) suggested that the use of self-promotion impression management (IM) tactics may positively influence job fit, but can backfire by making applicants seem insincere and manipulative once overuse is implemented. In recruitment software, such as the NLP- and ML-based resume analyzers (Sanyal et al., 2023; Daryania et al., 2021), keyword detection already identifies relevant skills. A Resume Keyword Highlighter aligns with this by emphasizing job-specific terms while allowing recruiters to distinguish between genuine qualifications and exaggerated self-promotion.

	A balanced self-promotion can often enhance evaluations of job fit and organizational fit (Chen & Lin, 2014; Kristof-Brown et al., 2002). Using keywords related to skills, experience, and achievements can signal competence that aligns with the recruiter’s expectations. For example, NLP-based resume screeners (Ofoegbu, 2024) showed strong accuracy in detecting relevant candidate skills and KSAs, helping reduce recruiter workload. A keyword highlighter applies the same principle by emphasizing self-promotional terms, ensuring that applicants highlight relevant skills while enabling recruiters to evaluate applications more efficiently.

	Excessive and obvious self-promotion may harm applicant credibility (Baron, 1986; Fletcher, 1989). Strong IM tactics may be seen as manipulative and insincere (Crant, 1996; Gurevitch, 1984) and may distort true applicant competence. Automated screening systems may possibly reinforce this problem if it rewards keyword-stuffing without providing context. A Resume Keyword Highlighter improves on this by not ranking candidates solely on repetition but visually distinguishing relevant keywords in context. This gives recruiters a clearer picture of whether the applicant is genuinely qualified or simply overusing self-promotional words.

2.4 Machine Learning in Resume Screening
In the modern era, machine learning has often been used in resume screening, particularly through the use of classifiers such as Naïve Bayes and K-Nearest Neighbors (KNN) for categorization and candidate ranking. Onukwugha, Ofoegbu, Aliche, and Betrand (2024) demonstrated that both algorithms achieved high accuracy in predicting resume effectiveness, with Naïve Bayes offering fast computation for large datasets and KNN providing strong performance in identifying similarities between candidate skills and job requirements. These ML methods are beneficial for small-scale projects like a Resume Keyword Highlighter, since they are time-efficient and can still provide reliable results in categorizing resumes. Integrating such modest approaches can improve recruiter efficiency and reduce workload without the complexity of commercial-level AI platforms.

2.5 AI-Driven Resume Screening and Bias Mitigation 

	
 	Navarro et al. (2025) introduced JustScreen, a resume screening system that enhances existing Applicant Tracking Systems (ATS) by applying Natural Language Processing (NLP), machine learning, and generative AI. Their framework includes preprocessing, skill extraction, fairness metrics, bias mitigation, and interpretability, allowing resumes to be assessed more accurately and equitably. Unlike conventional ATS tools that primarily filter applications, JustScreen emphasizes eliminating biases and promoting merit-based candidate selection.
By incorporating generative AI, JustScreen provides detailed resume analysis and actionable recommendations that add significant value to traditional ATS functionalities. Initial evaluations suggest that the system not only improves recruitment efficiency but also advances fairness and transparency in the hiring process.
This study underscores a shift in recruitment technologies from purely efficiency-focused automation toward ethical and unbiased candidate assessment. Compared to works like Patlolla et al. (2023), which focus on helping applicants structure their resumes for ATS compatibility, Navarro et al. highlight the employer-side challenges of fairness and bias mitigation, showing how both perspectives are critical to modern recruitment research.


2.6 A Survey on Bias and Fairness in Machine Learning

Mehrabi et al. (2022) provide context for research on fairness in artificial intelligence by showing how bias in data and algorithms can lead to unfair decisions in areas like hiring, criminal justice, and facial recognition. The survey points out that while AI systems can process large amounts of information quickly, they can still copy or even worsen human biases. It identifies gaps in knowledge, especially in understanding how bias can move between data, algorithms, and user behavior in a cycle that reinforces unfair outcomes. The authors also review different ways that fairness has been defined and studied, and they explain methods that have been tried in machine learning, deep learning, and natural language processing to reduce unfairness. This work supports the theoretical framework of fairness by organizing existing research into clear categories and examples. Finally, it justifies the need for more studies by showing that most current solutions focus only on technical fixes and do not fully deal with the larger social and ethical issues.

2.7 Automated Resume Parsing: Techniques and Challenges
Deepa et al. (2025) provided a comprehensive review of automated resume parsing approaches, ranging from traditional rule-based methods to advanced machine learning and deep learning techniques such as Named Entity Recognition (NER) and Transformers. Their study highlighted how these technologies enable the extraction, classification, and organization of candidate information across multiple resume formats, including PDF, DOCX, and scanned images.
They also identified key challenges in modern resume parsing, such as inconsistent data formatting, multilingual parsing, and ethical concerns in AI-driven hiring. To address these, the authors emphasized the importance of hybrid approaches that combine rule-based, ML, and deep learning techniques, as well as the use of evaluation metrics like precision, recall, and F1-score for assessing parser performance.
Finally, the study emphasized the need for future research on enhancing Applicant Tracking System (ATS) integration, improving fairness through bias mitigation, and advancing parsing models capable of handling complex layouts and unstructured resumes. Their work underscores the ongoing transition toward AI-powered recruitment tools that balance efficiency, accuracy, and ethical considerations.


2.8 Resume keywords

This article explains the importance of using strong and meaningful language in résumés and cover letters. It warns against using tired clichés, jargon, or weak words that can make an application look boring or unoriginal. Instead, it suggests using “power words” that are clear, specific, and action-oriented to show achievements, skills, and strengths. Employers often scan documents quickly, so every word should be chosen carefully to highlight accomplishments in a way that immediately catches attention. The guide also lists helpful modifiers, adjectives, and key traits to describe personality and soft skills, along with strong verbs to begin bullet points in the experience section. These words emphasize initiative, results, and impact. Overall, the message is that concise, active, and precise language makes a résumé stand out and creates a stronger impression on recruiters compared to vague or repetitive wording.

2.9 ATS Limitations and Talent Loss
Johnson (2023) reported that a significant percentage of employers—around 88%—acknowledged losing highly qualified candidates because of applicant tracking systems (ATS). These systems often screen out resumes that are not optimized for specific formats, even if the applicant possesses strong skills and relevant experience. This shows that the strict reliance on ATS may inadvertently prevent organizations from accessing top talent.
According to Allen, a professor of industrial and organizational psychology at the University of Utah, ATS are only as effective as their programming. When resumes fail to align with predefined templates or when nuanced skills are overlooked, highly capable candidates may be missed. The result is that top candidates may end up working for competitors that use less automated, more human-centered hiring approaches.
Experts like Miller and Kumar further emphasize the importance of monitoring ATS performance. They argue that HR professionals should adopt safeguards such as adverse incident management protocols and direct oversight to reduce bias and misclassification. While ATS undeniably improve efficiency and save time in large-scale recruitment, they lack human intuition, which means employers risk making costly errors by overlooking the most suitable candidates.
2.10 Enhancing Employability Through Resume-Writing Workshops
Zakaria et al.'s (2023) study explored how resume-writing workshops can improve students’ employability by helping them create more effective resumes. Using a survey of 26 participants from polytechnics and community colleges, the research found that most students were satisfied with the lessons and materials provided, with 77% stating the content was directly relevant to their needs and 69.23% finding the materials appropriate for their skill level. The use of authentic teaching materials, such as printed resources and word cards, enhanced learning and helped participants better understand how to craft strong resumes. However, the study was limited by its small sample size, narrow participant background, and short teaching sessions, which allowed little time for personalized feedback. The researchers suggest that future studies should include participants from different educational settings and extend lesson time to give students more tailored guidance. Overall, the workshop proved effective in improving students’ resume-writing skills and provided a useful model template for future job applications.



2.11 Graduate Employability Skills in the Philippine Context
Aguilar and Torres (2023) highlighted the evolving expectations of employers in both local and global markets, emphasizing the need for graduates to demonstrate a combination of skills, knowledge, and attitudes that define a “job-ready candidate.” Their study, conducted in a leading Philippine university, gathered insights from parents, alumni, and graduating students. Results identified social and communication skills, course-based competencies, and intrinsic motivation such as passion and commitment as the three most critical attributes. These findings underscore the importance of aligning higher education outcomes with labor market needs, suggesting that universities must re-evaluate graduate attributes to strengthen employability in a competitive global workforce.
2.12 Enhancing Employability Through Resume-Writing Workshops
This study focuses on the employability skills needed by graduates to succeed in the professional service sectors of Bonifacio Global City (BGC), a rapidly growing business hub in the Philippines. While technical knowledge is important, research shows that 85% of job success depends on well-developed soft skills such as communication, adaptability, and teamwork, with only 15% linked to technical skills. The study highlights how employers in BGC value a mix of both: technical proficiency to keep up with digital advancements, and strong soft skills like communication, problem-solving, critical thinking, and cross-cultural competence to thrive in fast-paced, multicultural environments.
The findings reveal that communication and interpersonal skills are the most essential for graduates working in BGC’s highly collaborative industries, while continuous learning, adaptability, and resilience are equally critical due to constant technological and organizational change. Ethical standards and business acumen are also emphasized, especially in fields like law and finance where trust and reputation matter.
Overall, this research shows that graduates, employers, and schools must work together to balance technical training with soft skills development. For graduates, this means focusing on both technical literacy and personal growth. For employers, it suggests creating recruitment and training programs aligned with these priorities. And for schools, it points to the importance of integrating communication, adaptability, and ethics into the curriculum to prepare students for BGC’s competitive and dynamic job market.






2.13 Digital Labor Market Intermediaries (DLMIs) in the Philippines
Businesses that act as platforms for connecting employers and employees are known as Labor Market Intermediaries (LMIs), and when these operate solely online, they are termed Digital Labor Market Intermediaries (DLMIs) (In the Philippines, DLMIs have become increasingly popular for both advertising employment (Autor, 2001).  opportunities and searching for job vacancies, reflecting the labor market’s adaptation to digital transformation. However, existing recruitment and placement policies under the Labor Code of the Philippines lack clarity in addressing the operations of these digital platforms.
Research highlights four key areas of DLMIs: (1) policies and legal frameworks that govern their activities, (2) services they provide to job seekers and employers, (3) their impact on job seekers’ pre-employment experiences, and (4) their effect on employers’ recruitment practices. While DLMIs improve accessibility and efficiency in labor matching, the absence of tailored policies raises concerns over data privacy, cybersecurity, and regulation.
For the present study, the role of DLMIs provides critical context: they illustrate how digital platforms have reshaped employment facilitation. Just as DLMIs bridge job seekers and employers at scale, the Resume Keyword Highlighter seeks to enhance applicant visibility within these systems. By ensuring that relevant skills and qualifications stand out in resumes, the tool supports applicants in navigating DLMIs more effectively, addressing one of the major challenges of digital hiring platforms—resumes being overlooked due to formatting or keyword mismatches.

2.14 Employability of IT Graduates in the Philippines
Albina and Sumagaysay (2020) conducted a tracer study of Information Technology (IT) and Computer Science graduates from a Philippine state university to assess their employment status, job relevance, and perceived usefulness of college competencies. They found that 78.53% of respondents were employed, and among them, 69.78% believed their first job was related to their chosen field of study. The study also revealed that communication skills and technical IT skills were among the most useful competencies for graduates in their early careers. 
While the study provides evidence that many graduates secure relevant employment, it also points to gaps in employability—particularly the time it takes some graduates to land their first job and the misalignment between curriculum and job demands.
This research supports the need for tools like the skill highlighter by demonstrating that graduates must not only possess relevant skills but also effectively communicate them in their résumés. Highlighting key competencies such as communication and IT skills can improve visibility in hiring systems, thus helping more graduates translate their college training into meaningful employment.



2.15 Keyword Optimization in Resumes for ATS
Placer (2024) underscores the importance of strategically incorporating relevant keywords into résumés to enhance their compatibility with Applicant Tracking Systems (ATS). ATS software scans résumés for specific keywords that match the job description, often related to required skills, qualifications, and experience. This process helps employers quickly narrow down the pool of applicants, making it essential for résumés to be tailored to each job application.

Placer (2024) provides several strategies for optimizing résumés, including analyzing job descriptions to identify relevant keywords, incorporating these keywords naturally throughout the résumé, and tailoring each résumé for individual job applications. Additionally, using synonyms, acronyms, and both long and short forms of terms can improve keyword coverage without compromising readability.

The article also emphasizes the importance of ATS-friendly formatting, such as using standard section headings, simple fonts, and text-based layouts, to prevent the system from misinterpreting or discarding information. Finally, Placer (2024) recommends testing résumés with online tools to ensure proper keyword integration and formatting.



2.16 Understanding ATS for Filipino Job Seekers
This article explains how Applicant Tracking Systems (ATS) affect job applications in the Philippines. ATS software helps companies manage large numbers of resumes by storing them, filtering them with keywords, and ranking applicants. Many Filipino employers, including SMEs and BPOs, use ATS to save time, ensure fair screening, and improve hiring efficiency.

The text also addresses several common misconceptions: ATS does not automatically reject 90% of resumes, small companies may also utilize these systems, submitting multiple applications for the same company can negatively affect candidates’ chances, simple formatting remains effective, and passing an ATS scan does not guarantee an interview.

Understanding ATS is important in the Philippines because the job market is very competitive, and labor laws encourage fair recruitment practices. To improve their chances, applicants should make ATS-friendly resumes: use simple formats, clear headings, job-related keywords, bullet points, and plain-text contact information.

2.17 Best ATS Providers in the Philippines
The article explains how Applicant Tracking Systems (ATS) are changing hiring practices in the Philippines, especially in industries like BPO, IT, healthcare, and finance, where employers receive large volumes of resumes. An ATS is software that helps companies manage applications, filter resumes using keywords, schedule interviews, and generate hiring reports. It improves efficiency, ensures fairness in screening, and keeps all candidate information organized in one place.

The article highlights why Filipino businesses need ATS: it saves time, improves the quality of hires, helps teams collaborate, supports remote hiring, and ensures compliance with local labor laws. It also lists must-have ATS features such as AI-powered candidate matching, resume parsing, job board integration, and data privacy protections.

Transformify is presented as one of the top ATS providers, offering AI-driven recruitment, one-click job postings to over 100 sites, advanced data analytics, and integrations with major HR and payroll systems. The article also compares other ATS providers like iSmartRecruit, 100Hires, JobDiva, Crelate, and Kalibrr, each catering to different company sizes and industries.

Finally, it discusses 2025 ATS trends in the Philippines, including the rise of AI and automation, mobile-first hiring, cloud-based platforms, diversity-focused recruitment, and integrated payment systems. The key takeaway is that using ATS helps businesses in the Philippines hire faster, smarter, and more fairly, while also adapting to the growing demand for digital and remote recruitment.



2.18 Fair and Transparent AI-Driven Resume Screening
The study “Fair and Transparent AI-Driven Resume Screening: Enhancing Recruitment with Bias-Aware Machine Learning” focuses on improving the resume screening process using artificial intelligence (AI) and machine learning (ML). Traditional resume screening is time-consuming, often biased, and may overlook qualified candidates with unconventional backgrounds. To solve these issues, the researchers developed an AI-driven system that uses natural language processing (NLP) tools like TF-IDF, Word2Vec, and BERT embeddings, along with classifiers such as Random Forest and SVM. The system achieved 93% accuracy and cut processing time per resume from 3.1 seconds to 1.2 seconds, making hiring faster and more precise.

A key feature of the system is its fairness-aware algorithms, which reduced demographic bias by about 30%, ensuring more equitable hiring. The model included bias monitoring dashboards to continuously check and adjust decisions, which increased transparency and compliance with ethical AI standards. However, challenges remain, such as computational costs of transformer models like BERT, and subtle biases still present in historical training data.
The paper suggests future improvements like better bias-mitigation techniques, explainable AI (XAI) methods to clarify decisions, scalability enhancements, and multimodal candidate evaluations (including video or voice). It also highlights the need for collaboration between AI and human recruiters, as well as improving candidate experience through feedback and personalization.

Overall, the research shows that AI-powered resume screening can significantly improve accuracy, fairness, and efficiency in recruitment, but ongoing refinements are necessary to fully ensure ethical and transparent hiring practices.


2.19 Hidden Workers: Untapped Talent
The 2021 “Hidden Workers” study by Harvard Business School and Accenture is the largest independent look at how automated hiring filters affect real labor-market outcomes. Surveying 2,275 hiring executives and 8,720 people who have been shut out of the workforce, the researchers found that the very tools adopted to speed up recruiting have become its biggest bottleneck. Eighty-eight percent of employers agreed their applicant-tracking systems (ATS) routinely eliminate high-skill candidates who could perform well on the job; for middle-skill roles the figure rises to 94 %. These systems rely on exact-keyword matches, uninterrupted employment histories, and college-degree proxies, automatically rejecting résumés that deviate even slightly from the template.

Because the filters are so rigid, an estimated 27 million U.S. adults—caregivers, veterans, career-changers, neuro-diverse applicants, and others—remain “hidden” from recruiters although they possess the necessary competencies. The report shows that when companies deliberately widen the aperture—relaxing keyword rules, accepting skills evidence in varied language, and assessing competencies rather than credentials—they fill vacancies faster and outperform their peers. Firms that actively hire hidden workers report being 36 % less likely to face talent shortages and rate those employees equal or superior to traditional hires on attitude, productivity, attendance, and innovation.

Ultimately, the study argues that maximizing efficiency inside the ATS is a false economy: it externalizes the cost of unfilled jobs and shrinks the talent pool. Reversing the logic—rewriting job descriptions with must-have skills, switching from “negative” to “affirmative” filters, and rewarding recruiters for quality-of-hire rather than cost-per-hire—restores access to a large, motivated, and demonstrably capable segment of the labor force.


2.20 Synthesis
	The reviewed literature highlights the ongoing transformation of recruitment practices, shaped largely by automation, artificial intelligence, and the need for fairness in candidate evaluation.

Several studies (Daryani et al., 2020; Sanyal et al., 2017; Patlolla et al., 2023) emphasized the importance of resume parsing and keyword extraction, which serve as the foundation for automated screening tools. These methods are useful in identifying skills and matching candidates to job descriptions, but they also risk reducing applicant evaluation to surface-level keyword frequency. Research on self-promotion in resumes (Godfrey et al., 1986; Waung et al., 2016) further shows how applicants strategically use keywords, but excessive keyword-stuffing can undermine credibility. Thus, automated tools must balance keyword detection with contextual understanding.

Machine-learning methods (Onukwugha et al., 2024) and advanced NLP-based systems (Navarro, 2025; Shah et al., 2025) push this further by integrating bias-mitigation techniques, explainable AI, and fairness metrics into screening. These studies highlight the risks of biased decision-making when systems rely on historical hiring data, but they also point to promising solutions such as fairness-aware algorithms and monitoring dashboards. At the same time, Mehrabi et al. (2022) cautioned that AI may still reproduce social biases, reinforcing the importance of ethical frameworks alongside technical improvements.

Other studies explored context-specific factors: Johnson (2023) and Placer (2024) warned that strict ATS filters can cause organizations to lose top talent if résumés are not optimized with proper keywords. Local research (Aguilar & Torres, 2023; Albina & Sumagaysay, 2020; Aguenza & Ingles, 2024) showed that employability in the Philippines depends not only on technical competencies but also on soft skills like communication, adaptability, and professionalism. Studies on resume-writing workshops (Zakaria et al., 2023) demonstrated that guided support helps applicants prepare better résumés, while digital labor market intermediaries (Pineda & Cayabyab, 2023) illustrate the increasing role of online platforms in shaping how applications are processed.
Taken together, these studies underline a shared challenge: applicants must present their skills effectively, while employers must screen applications fairly and efficiently. Current ATS and AI-driven systems have advanced in accuracy and fairness, but gaps remain in contextual understanding, bias reduction, and applicant empowerment. The 2021 Harvard Business School and Accenture “Hidden Workers” report adds that rigid ATS rules hide an estimated 27 million qualified candidates, suggesting that relaxing keyword filters and assessing competencies rather than credentials can restore access to valuable talent.

The proposed system, SkillHighlight, builds on these insights by applying BERT to not just detect but contextually highlight meaningful keywords in résumés. Unlike systems that rely on frequency counts or rigid keyword matching, SkillHighlight is designed to help applicants emphasize genuine skills while assisting recruiters in identifying relevant qualifications. By combining the strengths of keyword optimization, bias-aware AI, and contextual NLP, this study aims to contribute to more effective and equitable recruitment processes.


2.21 Theoretical Framework
The foundation of this study is based on the ideas and findings presented in previous research related to resume optimization, employability, and recruitment technologies. Several studies (Sanyal et al., 2017; Daryania et al., 2020; Patlolla et al., 2023) emphasize the importance of parsing and keyword extraction as tools for identifying essential skills and qualifications in resumes. These works demonstrate how Natural Language Processing (NLP) and Machine Learning (ML) improve the accuracy and efficiency of matching candidates to job descriptions.

Studies on employability (Aguilar & Torres, 2023; Albina & Sumagaysay, 2020) highlight the need for applicants to effectively communicate their skills, competencies, and experiences to improve their chances of employment. This supports the development of systems that enhance visibility of job-relevant information. Similarly, research on Applicant Tracking Systems (Johnson, 2023; Navarro, 2025) identifies issues such as bias, unfair filtering, and loss of qualified candidates due to poor resume formatting or lack of keyword optimization.

Together, these studies form the theoretical foundation of SkillHighlight. They establish that the ability to extract, highlight, and present relevant keywords in resumes not only streamlines recruitment but also promotes fairness and better candidate-employer matching. By combining principles from resume parsing, keyword extraction, and employability enhancement, this system aims to assist both applicants and recruiters in improving the overall hiring process.

2.22 Conceptual Framework
	
The study, titled “A BERT-Based Approach to Context-Aware Keyword Highlighting in Résumé Screening,” introduces SkillHighlight, an intelligent résumé analysis system that evaluates professional language using Natural Language Processing (NLP) and Machine Learning (ML). The framework illustrates how input résumé text is transformed through BERT-based semantic modeling, deep-learning classification, and keyword pattern recognition into meaningful analytical outputs that support résumé enhancement and screening.

A. Input
Résumé Text Data
 The system accepts résumé content as input, either through file upload (PDF, DOCX, TXT) or direct text entry. This textual data serves as the primary material for linguistic and contextual analysis.


Keyword Databases
 The system utilizes four predefined keyword categories to guide the analysis:
 a. Action Verbs – express initiative, leadership, and measurable achievement.
 b. Soft Skills – indicate interpersonal, cognitive, and communication abilities.
 c. Hard Skills – refer to technical and domain-specific competencies.
 d. Recruiter Keywords – represent high-impact terms commonly found in job descriptions and professional profiles.

 These inputs establish the foundational elements for automated résumé evaluation.


B. Process
 Text Preprocessing
SpaCy performs tokenization, part-of-speech tagging, and segmentation, enhanced by a custom smart-split algorithm optimized for résumé structure (e.g., bullet points, section headings).
 Contextual Embedding via BERT
 SkillHighlight uses BERT to encode each sentence into contextual embeddings, enabling interpretation of meaning based on linguistic context rather than keyword frequency alone.


Self-Promotion Classification (BERT + KNN)
 Each sentence is converted into embeddings using BERT (all-MiniLM-L6-v2), then  classified using a K-Nearest Neighbors (KNN) model trained on labeled resume data. The prediction determines the assertiveness and achievement-oriented strength of the sentence. Sentiment polarity and metric-based heuristics provide secondary boosts to ensure that measurable accomplishments and positive language are rewarded appropriately.


 Keyword Recognition and Highlighting
 The system identifies matching skills across the four categories and highlights them in color for intuitive visualization.


 Analytical Visualization
 The system produces percentage-based skill composition, an overall self-promotion score, and sentence-level evaluations that guide résumé improvement.


C. Output
Output
Description
1. Highlighted Résumé Text
Color-coded skill visualization.
2. Self-Promotion Score
Numeric evaluation of assertive achievement-based language.
3. Skill Composition Metrics
Percentage of hard, soft, and recruiter-keywords.
4. Sentence-Level Insights
Scored feedback to identify weak and strong résumé statements

D. Evaluation
Accuracy Assessment
 Validates correct keyword detection and sentence scoring


Contextual Relevance
 Ensures highlighting aligns with intended meaning rather than raw frequency


User Perception
 Measures clarity, usability, and practical benefit to students and recruiters


E. Purpose of the Framework
The conceptual framework illustrates how SkillHighlight applies BERT-based semantic modeling and KNN classification to assess résumé language more accurately and fairly. BERT converts sentences into embeddings, which KNN evaluates to score self-promotion, supported by sentiment and metric heuristics. Visual keyword highlighting provides immediate feedback on strengths and areas for improvement. Overall, the framework enhances résumé clarity and screening efficiency for both applicants and recruiters, with evaluation guiding continuous system refinement.


Figure 2. Conceptual Framework of the Study
	The diagram illustrates how the SkillHighlight system converts résumé text input into analytical feedback through NLP and machine-learning processes, with an evaluation component providing feedback for system refinement.