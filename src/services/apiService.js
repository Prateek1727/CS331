const API_BASE_URL = 'http://localhost:8000/api';

export const ticketService = {
  /**
   * Fetch all tickets from the backend.
   */
  async getTickets() {
    try {
      const response = await fetch(`${API_BASE_URL}/tickets`);
      if (!response.ok) throw new Error('Failed to fetch tickets');
      return await response.json();
    } catch (error) {
      console.error('API Error fetching tickets:', error);
      return [];
    }
  },

  /**
   * Submit a new raw ticket to be processed by the AI engine.
   */
  async createTicket(ticketData) {
    try {
      const response = await fetch(`${API_BASE_URL}/tickets`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(ticketData),
      });
      if (!response.ok) throw new Error('Failed to create ticket');
      return await response.json();
    } catch (error) {
      console.error('API Error creating ticket:', error);
      throw error;
    }
  },

  /**
   * Submit a ticket with image upload for fraud detection.
   */
  async createTicketWithImage(ticketData, imageFile) {
    try {
      const formData = new FormData();
      formData.append('channel', ticketData.channel);
      formData.append('customer_name', ticketData.customer_name);
      formData.append('customer_email', ticketData.customer_email);
      formData.append('subject', ticketData.subject);
      formData.append('message', ticketData.message);
      
      if (imageFile) {
        formData.append('image', imageFile);
      }

      console.log('Sending request to:', `${API_BASE_URL}/tickets/upload`);
      
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 second timeout

      const response = await fetch(`${API_BASE_URL}/tickets/upload`, {
        method: 'POST',
        body: formData,
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Server error:', errorText);
        throw new Error(`Failed to create ticket with image: ${response.status} ${errorText}`);
      }
      return await response.json();
    } catch (error) {
      console.error('API Error creating ticket with image:', error);
      if (error.name === 'AbortError') {
        throw new Error('Request timeout - AI analysis took too long');
      }
      throw error;
    }
  },

  /**
   * Approve ticket and send notifications (PRODUCTION)
   */
  async approveTicket(ticketId, agentId = 'AGENT-001') {
    try {
      const response = await fetch(`${API_BASE_URL}/actions/approve`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ ticket_id: ticketId, agent_id: agentId }),
      });
      if (!response.ok) throw new Error('Failed to approve ticket');
      return await response.json();
    } catch (error) {
      console.error('API Error approving ticket:', error);
      throw error;
    }
  },

  /**
   * Update draft response (PRODUCTION)
   */
  async updateDraft(ticketId, message, agentId = 'AGENT-001') {
    try {
      const response = await fetch(`${API_BASE_URL}/actions/update-draft`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ ticket_id: ticketId, message, agent_id: agentId }),
      });
      if (!response.ok) throw new Error('Failed to update draft');
      return await response.json();
    } catch (error) {
      console.error('API Error updating draft:', error);
      throw error;
    }
  },

  /**
   * Escalate ticket (PRODUCTION)
   */
  async escalateTicket(ticketId, reason, agentId = 'AGENT-001') {
    try {
      const response = await fetch(`${API_BASE_URL}/actions/escalate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ ticket_id: ticketId, reason, agent_id: agentId }),
      });
      if (!response.ok) throw new Error('Failed to escalate ticket');
      return await response.json();
    } catch (error) {
      console.error('API Error escalating ticket:', error);
      throw error;
    }
  },

  /**
   * Reject ticket (PRODUCTION)
   */
  async rejectTicket(ticketId, reason, agentId = 'AGENT-001') {
    try {
      const response = await fetch(`${API_BASE_URL}/actions/reject`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ ticket_id: ticketId, reason, agent_id: agentId }),
      });
      if (!response.ok) throw new Error('Failed to reject ticket');
      return await response.json();
    } catch (error) {
      console.error('API Error rejecting ticket:', error);
      throw error;
    }
  },

  /**
   * Get action logs for a ticket
   */
  async getActionLogs(ticketId) {
    try {
      const response = await fetch(`${API_BASE_URL}/actions/logs/${ticketId}`);
      if (!response.ok) throw new Error('Failed to fetch action logs');
      return await response.json();
    } catch (error) {
      console.error('API Error fetching action logs:', error);
      return { logs: [] };
    }
  }
};
