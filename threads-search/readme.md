## 1. Create a Threads App on Meta for Developers
   - Visit the Meta Developer platform: [Meta for Developers](https://developers.facebook.com/apps)
   - Create a new app to interact with the Threads API.

## 2. Enable Required Functions
   - In your newly created Threads app, **enable the following functions**:
     - `threads_basic`
     - `threads_keyword_search`
     - `threads_profile_discovery`

## 3. Add API Credentials to `.env`
   - Copy your **Threads App ID** and **Threads App Secret** (code) from the Meta Developer console.
   - Open your `.env` file and add the following lines:
     ```env
     API_ID=your_app_id
     API_SECRET=your_app_secret
     ```

## 4. Add Test User to App Role
   - Add a **test user** to the **Threads App** role in the developer console.
   - The test user must agree to the terms.  
   - Important Note: If you want to obtain information about all members, your app needs to pass the review process by Meta. If it hasn't been reviewed, you will only be able to retrieve information about the authenticated user.

## 5. Download and Install Ngrok
   - Download Ngrok based on your operating system from [Ngrok Download Guide](https://ngrok.com/download).
   - Follow the instructions to install it on your machine.

## 6. Install and Configure Ngrok
   - After downloading Ngrok, run the following commands to install and configure it:
     1. Unzip the downloaded Ngrok file:
        ```bash
        sudo unzip ~/Downloads/ngrok-v3-stable-darwin-arm64.zip -d /usr/local/bin
        ```
        *(Adjust the path based on your operating system)*
     2. Add your Ngrok authentication token:
        ```bash
        ngrok config add-authtoken YOUR_AUTH_TOKEN
        ```
        *(You can find your **AUTH_TOKEN** after logging into Ngrok.)*

## 7. Navigate to the Project Directory
   - Open the terminal and navigate to your project directory:
     ```bash
     cd /Users/fantastic-lin/Documents/part-time_job/threads-search/
     ```
     *(Replace this path with your actual project directory.)*

## 8. Create a Virtual Environment
   - Create a virtual environment named `threads_search` by running the following command:
     ```bash
     conda env create -f environment.yml
     ```

## 9. Activate the Virtual Environment
   - Activate the newly created virtual environment:
     ```bash
     conda activate threads_search
     ```

## 10. Run the Application
   - Run the application to start the Threads API integration:
     ```bash
     python app/threads_app.py
     ```

## 11. Expose the Local Server Using Ngrok
   - Open a new terminal and run Ngrok to expose your local server (replace `8443` with the port set in `.env` if necessary):
     ```bash
     ngrok http 8443
     ```
     *(Optional: You can specify a custom URL by adding `--url=<your_custom_url>`.)*

## 12. Update the Redirect URI in Meta Developer Settings
   - After running the Ngrok command, you'll get a forwarding URL, for example:
     ```
     https://unhealthier-nonpacifistic-darell.ngrok-free.dev
     ```
   - Add `/callback` to the end of this URL and use it in your Meta Developer app settings:
     ```text
     https://unhealthier-nonpacifistic-darell.ngrok-free.dev/callback
     ```
   - Update the **Redirect URI**, **Uninstall callback URL**, and **Delete callback URL** with this link.

## 13. Obtain SMTP Credentials for Email Notifications
   - If you wish to send automatic email notifications, you'll need SMTP credentials.
   - Obtain the **SMTP_USER** and **SMTP_PASS** from your email provider.
   - Add these credentials to your `.env` file:
     ```env
     SMTP_USER=your_email@example.com
     SMTP_PASS=your_smtp_password
     ```

## 14. Verify the Application
   - Open the Ngrok URL in your browser (e.g., `https://unhealthier-nonpacifistic-darell.ngrok-free.dev`).
   - After verifying, the page should display:
     ```
     Authorization successful. ✅ You can close this page now.
     ```

## 15. Using the Application
   - Now, you can perform searches using the following commands:

### (1) **Keyword Search**
   - To search for keywords (e.g., "hello world"), run the following command:
     ```bash
     python ./scripts/run_search.py keyword hello world --termfreq_mode --send_email
     ```

### (2) **Username Search**
   - To search by username (e.g., "meta" or "Facebook"), use the following command:
     ```bash
     python ./scripts/run_search.py username meta Facebook --termfreq_mode --send_email
     ```

---

### Additional Notes:
- **`--termfreq_mode`**: This option is used to count word frequency in the search results.
- **`--send_email`**: This option sends an email with the search results.
- Ensure that **SMTP credentials** are correctly set up to use the email functionality.

---

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
