# Software Requirements Specification (SRS)

## 1. Introduction

### 1.1 Purpose
This document specifies the functional and non‑functional requirements for the Agriculture Advisory System (AAS), a web-based application that guides farmers and agricultural stakeholders. It defines what the system will deliver, the operating constraints, and the metrics used to verify compliance. The SRS serves as a contract between the development team and the project sponsors, ensuring alignment on scope and behaviour.

The system focuses on providing actionable advice rather than raw data; for example, disease prediction results are accompanied by treatment suggestions, and weather forecasts include planting recommendations.


### 1.2 Scope
The product is a standalone Flask‑based web portal. Key modules include user management, crop database, disease detection via ML, weather and mandi rate display, forums and informational pages.

The system will be delivered as a single‑page style application with server‑rendered templates. No mobile clients or REST API are provided in the initial release. Data is stored in flat JSON files; no relational database or persistent cache is used. The application is intended for use in rural and semi‑urban areas where network bandwidth may be limited.

**In scope**:
- Registration/login and role‑based access.
- Browsing crop information filtered by state.
- Image upload and disease inference using a trained neural network.
- Displaying current and forecast weather conditions.
- Listing commodity prices from mandi markets.
- Discussion forum and static advisory pages.

**Out of scope**:
- Automated SMS or voice alerts.
- Payment or e‑commerce functionality.
- Multi‑language support in the first version.
### 1.3 Definitions, Acronyms, Abbreviations
- **SRS**: Software Requirements Specification
- **UI**: User Interface
- **API**: Application Programming Interface
- **ML**: Machine Learning
- **JSON**: JavaScript Object Notation

### 1.4 References
- `app.py` – Flask routes and application logic
- JSON resources in `model/`
- Templates under `templates/` and static assets

### 1.5 Document Conventions
- **Shall** indicates a requirement that must be met.
- **Should** indicates a recommendation that has significant benefits but is not mandatory.
- All identifiers such as FR‑xxx refer to functional requirements listed in section 3.
- Code fragments, file names and user interface elements are rendered in `monospace`.

## 2. Overall Description

### 2.1 Product Perspective
The system is a self‑contained web application. It uses Python/Flask, HTML/CSS/JS templates, a trained ML model for leaf disease classification and JSON data for crops and diseases.

### 2.2 User Classes and Characteristics
| User Class | Description | Primary Goals | Permissions |
|------------|-------------|---------------|-------------|
| Farmer | End user cultivating crops | Obtain crop advice, diagnose plant health, view weather/market data | Browse data, post/read forum, upload images |
| Extension Officer | Agricultural expert or government representative | Provide accurate information, moderate content | Edit crop/disease entries, moderate forum posts |
| Administrator | Technical maintainer | Manage user accounts, system configuration | Full access including server settings |

Each user class interacts with the system via a standard web browser; no specialized training is required. Farmers and officers may use the platform on mobile devices, hence the UI must adapt to smaller screens.
### 2.3 Operating Environment
- **Server**: Python 3.8 or above installed; Flask framework and dependencies listed in `requirements.txt`.
- **Hosting**: Can run on Windows or Linux; typical deployment targets include Heroku, AWS EC2, or a local VM.
- **Client**: Modern web browser (Chrome, Firefox, Edge, Safari) with JavaScript and cookies enabled. Supports desktop and mobile form factors.
- **Network**: Internet connection required for external API calls; the site should function with bandwidth as low as 256 kbps.

### 2.4 Design and Implementation Constraints
- Data stored in JSON files; no relational database. The design must include an abstraction layer to allow future migration to a database without rewriting business logic.
- Model files residing in `model/` folder; loading must occur at server start and be thread‑safe for concurrent requests.
- The application must use only open‑source libraries compatible with a permissive license (MIT/BSD).
- File operations must handle concurrent read/write scenarios gracefully (e.g., using file locks or atomic writes).


### 2.5 Assumptions and Dependencies
- Internet availability for weather and mandi rate APIs; fallback data will be used if the network is unavailable.
- Users will upload clear, well‑lit images of leaves for disease detection; extremely poor images may lead to incorrect predictions.
- Browser compatibility with modern standards (HTML5, CSS3, ES6); legacy browsers (e.g., IE11) are not supported.
- Python packages listed in `requirements.txt` will be installable in the target environment.
- The server has sufficient memory (minimum 1 GB) to load the ML model and handle web requests.

## 3. System Features (Functional Requirements)

Each requirement is identified by a unique ID.

### 3.1 USER AUTHENTICATION (FR-001)
- **Description**: Provide account creation, login and logout functionality with session management.
- **Preconditions**: User must provide a unique and valid email address.
- **Inputs**: Email address, password (minimum 8 characters), optional name.
- **Process**: Validate inputs on client and server. Passwords are hashed (bcrypt) before storage in `users.json`. On login, credentials are compared securely.
- **Outputs**: Successful authentication sets a session cookie and redirects user to dashboard; failures return user-friendly error messages without revealing sensitive information.
- **Postconditions**: Authenticated users have access to protected features; inactive sessions expire after 30 minutes.
- **Error Handling**: Duplicate registration attempts yield a message "Email already registered." Invalid credentials show a generic "Login failed".


### 3.2 CROP INFORMATION (FR-002)
- **Description**: Allow users to browse and search a catalog of crops, filtered by state and category.
- **Inputs**: State selection from drop‑down, optional text search term.
- **Process**: Load `state_crops.json`, apply filters and/or search regex, paginate results if necessary. Each crop entry links to a detailed page with cultivation tips and images.
- **Outputs**: Rendered HTML listing matching crops. If no results, display a message encouraging a broader search.
- **Postconditions**: None; read‑only operation.
- **Error Handling**: If the JSON file is unreadable, log error and display "Information currently unavailable." Data changes require admin update.


### 3.3 DISEASE DETECTION (FR-003)
- **Description**: Accept an uploaded image of a plant leaf, classify it using the ML model and provide corresponding treatment advice.
- **Preconditions**: User must be logged in. Image must be in JPEG/PNG format and under 5 MB.
- **Inputs**: Image file, optionally crop type for context.
- **Process**:
   1. Validate image size and type.
   2. Pre‑process (resize, normalize) for model input.
   3. Run inference with the loaded TensorFlow/Keras model.
   4. Map predicted label to information stored in `crop_diseases.json`.
- **Outputs**: Results page with disease name, confidence score, description, and treatment recommendations.
- **Postconditions**: Prediction logged (optionally) for analytics. No state changes in data files unless admin updates remedies.
- **Error Handling**: Format/size violations prompt user to upload a valid image. Model errors result in a retry prompt with apology message.


### 3.4 WEATHER FORECAST (FR-004)
- **Description**: Retrieve and display the current weather conditions and a short‑term forecast for a user‑specified location.
- **Inputs**: Location name or geographic coordinates entered by the user.
- **Process**: Invoke external weather API (e.g., OpenWeatherMap), parse JSON response, extract parameters (temperature, humidity, wind, precipitation). Cache response for 10 minutes to reduce calls.
- **Outputs**: HTML fragment showing numerical data and iconography representing conditions. Includes advisory text such as "Suitable for sowing" or "Avoid spraying pesticides today" when applicable.
- **Error Handling**: API failures fall back to the most recent cached data. If none available, display "Weather data currently unavailable." Log API errors.


### 3.5 MANDI RATES (FR-005)
- **Description**: Provide users with up‑to‑date wholesale commodity prices from various regional markets.
- **Inputs**: Commodity selection, optional date range or market filter.
- **Process**: Fetch data from an external API or load from local JSON if offline. Sort and format the data into a tabular view. Prices are timestamped to indicate freshness.
- **Outputs**: Table view listing mandi names, commodity prices per unit, and last updated time.
- **Error Handling**: If the data source is unreachable, inform the user and show the last known values with an ‘‘as of’’ timestamp.


### 3.6 FORUM & INFORMATION PAGES (FR-006)
- **Description**: Enable registered users to post questions, reply to threads, and browse static informational pages covering topics such as government schemes, soil health, and best practices.
- **Inputs**: Post title, body text, optional image attachment.
- **Process**: Save new posts to a JSON-based store or database stub. Render threads in chronological order with pagination. Informational pages are rendered statically from templates.
- **Outputs**: Discussion threads visible to all users; officers/administrators can delete inappropriate content.
- **Error Handling**: Inputs are sanitized to prevent XSS; attachments exceeding size limits are rejected with an explanatory message.

### 3.7 ADMINISTRATION (FR-007)
- **Description**: A restricted interface allowing administrators to manage user accounts, update JSON data files (crops, diseases), and moderate forum content.
- **Inputs**: Admin credentials, uploaded JSON files, moderation actions (approve/delete posts).
- **Process**: Authenticate using role information. Expose CRUD forms for editing data sets. Write changes to disk with backups.
- **Outputs**: Immediate reflection of data updates across user-facing pages; audit logs of admin actions for accountability.
- **Error Handling**: Validation of JSON schema on uploads; invalid files are rejected with error details.

## 4. External Interface Requirements

### 4.1 User Interfaces
- HTML templates (`base.html`, page-specific templates) responsive across devices.
- CSS/JS under `static/` for styling and interactivity.

### 4.2 Hardware Interfaces
- No specialized hardware; optionally camera for image capture.

### 4.3 Software Interfaces
- Flask routes in `app.py` handling HTTP requests.
- Integration with third‑party APIs for weather or market data.

### 4.4 Communication Interfaces
- HTTP/HTTPS for client–server communication.
- JSON for data interchange.

### 4.5 Data/Database Interface
- Although persistent storage currently uses flat JSON files on the server, the application shall expose an abstraction layer (repository pattern) to allow a future migration to a relational (e.g., PostgreSQL) or NoSQL (e.g., MongoDB) database without significant rewrites.
- The layer provides methods such as `get_user(id)`, `save_crop(data)`, and `query_forum_posts(filter)`.

## 5. Non‑functional Requirements

### 5.1 Performance
| Requirement | Metric |
|-------------|--------|
| Page load time | ≤2 s on a 3G connection |
| Disease detection inference time | ≤5 s per request |
| Concurrent active sessions (initial) | Support 100 users without degradation |
| Data update latency | New crop/disease entries reflected within 60 s after admin edit |

The system shall degrade gracefully under load, returning a "Service busy, please try again" message when the session limit is reached.

### 5.2 Security
- All traffic must use HTTPS; HTTP requests shall redirect automatically.
- Passwords shall be hashed with a strong algorithm (e.g., bcrypt) and salted.
- Session cookies must be marked HttpOnly and Secure and expire after 30 minutes of inactivity.
- Role-based access control shall restrict administrative and officer functions.
- Input validation and output encoding shall prevent XSS, CSRF, and injection vulnerabilities.
- Sensitive configuration data (API keys, secrets) must be stored in environment variables, not source control.

### 5.3 Usability
- Simple, intuitive navigation; mobile‑friendly layout.

### 5.4 Reliability and Availability
- Target 99% uptime; error handling with meaningful messages.

### 5.5 Maintainability
- Well‑documented code; modular templates and static assets.
- JSON resources easily updatable.

### 5.6 Portability
- Deployable on common servers; compatible with Windows and Linux.

## 6. Other Requirements

### 6.1 Data Storage and Backup
- JSON files under version control; periodic backups of `model/` and templates.

### 6.2 Legal and Regulatory
- Comply with data privacy guidelines, especially for user information.
- Adhere to relevant agricultural advisory standards.

### 6.3 Future Enhancements
- Add database support
- Mobile app integration

## 7. Appendices

### A. Glossary
- **Mandi**: A marketplace where agricultural commodities are bought and sold.
- **Extension Officer**: A government or organizational representative who advises farmers.
- **JSON**: JavaScript Object Notation, a text format for data exchange.

### B. Change Log
| Date | Author | Description |
|------|--------|-------------|
| 2026-02-18 | Initial draft | Basic structure created |
| 2026-02-18 | Expanded content | Added detailed requirements and tables |

---

*Document prepared on February 18, 2026*